import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from tqdm import tqdm
import os
from PIL import Image
import io
import argparse
from azure.storage.blob import BlobServiceClient
import multiprocessing

# Allow loading of large images that might otherwise raise an error
Image.MAX_IMAGE_PIXELS = None

# --- Worker function for parallel processing ---
def process_blob(args):
    """
    Worker function: downloads, splits, and saves patches for a single image blob.
    """
    blob_name, connection_string, source_container, local_target_dir, patch_size = args
    
    # Each worker needs its own BlobServiceClient instance.
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(source_container)

    try:
        downloader = container_client.get_blob_client(blob_name).download_blob()
        image_bytes = downloader.readall()

        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            patch_num = 0
            for y in range(0, height - patch_size + 1, patch_size):
                for x in range(0, width - patch_size + 1, patch_size):
                    box = (x, y, x + patch_size, y + patch_size)
                    patch = img.crop(box).convert('RGB')

                    original_filename = os.path.splitext(blob_name)[0].replace("/", "_")
                    patch_filename = f"{original_filename}_patch_{patch_num}.png"
                    save_path = os.path.join(local_target_dir, patch_filename)
                    
                    patch.save(save_path)
                    patch_num += 1
        return f"Processed {blob_name}"
    except Exception as e:
        return f"Failed to process {blob_name}: {e}"

# --- The preprocessing function orchestrates the parallel work ---
def preprocess_and_save_locally(connection_string, source_container, local_target_dir, patch_size=224):
    """
    Orchestrates the parallel preprocessing of images from Azure.
    """
    if os.path.exists(local_target_dir) and len(os.listdir(local_target_dir)) > 0:
        print(f"Patches already exist in local cache ('{local_target_dir}'). Skipping preprocessing.")
        return

    print(f"Starting parallel preprocessing: downloading from '{source_container}' and saving patches to '{local_target_dir}'...")
    os.makedirs(local_target_dir, exist_ok=True)

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(source_container)
        image_blobs = [blob.name for blob in container_client.list_blobs() if blob.name.lower().endswith(('jpg', 'jpeg', 'png'))]

        if not image_blobs:
            print(f"Warning: No images found in source container '{source_container}'.")
            return

        # Prepare arguments for each worker process
        tasks = [(blob_name, connection_string, source_container, local_target_dir, patch_size) for blob_name in image_blobs]

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            # Use tqdm to show a progress bar for the parallel tasks
            for result in tqdm(pool.imap_unordered(process_blob, tasks), total=len(tasks), desc="Processing Images in Parallel"):
                pass
        
        print(f"✅ Parallel preprocessing complete. All patches saved to '{local_target_dir}'.")

    except Exception as e:
        print(f"FATAL: An error occurred during preprocessing orchestration: {e}")
        raise

class PatchedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        patch = Image.open(image_path).convert('RGB')
        
        if self.transform:
            # Apply the same transform twice to get two different augmented views
            patch1 = self.transform(patch)
            patch2 = self.transform(patch)
            return patch1, patch2
            
        return patch, patch

# Simple SimCLR-style contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features1, features2):
        batch_size = features1.shape[0]
        
        # Normalize features
        features1 = nn.functional.normalize(features1, dim=1)
        features2 = nn.functional.normalize(features2, dim=1)
        
        # Concatenate features
        features = torch.cat([features1, features2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)
        
        # Create mask for positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        mask = mask.float()
        
        # Create labels for contrastive loss
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0)
        labels = labels.to(features.device)
        
        # Compute logits
        similarity_matrix = similarity_matrix / self.temperature
        
        # Mask out self-similarity
        similarity_matrix.masked_fill_(torch.eye(2 * batch_size, dtype=torch.bool, device=features.device), -9e15)
        
        # Compute loss
        loss = self.criterion(similarity_matrix, labels)
        
        return loss

def main():
    SOURCE_DATA_CONTAINER = "data"
    LOCAL_PATCH_CACHE_DIR = '/mnt/data' 
    LOCAL_MODEL_OUTPUT_DIR = '/mnt/satellite-resnet'

    PATCH_SIZE = 224
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATALOADER_WORKERS = 4 if torch.cuda.is_available() else 0
    LR = 1e-3

    connection_string = "DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net"

    print(f"Using device: {DEVICE}")
    
    preprocess_and_save_locally(
        connection_string=connection_string,
        source_container=SOURCE_DATA_CONTAINER,
        local_target_dir=LOCAL_PATCH_CACHE_DIR,
        patch_size=PATCH_SIZE
    )

    try:
        # Create augmentation pipeline
        transform = T.Compose([
            T.RandomResizedCrop(size=PATCH_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = PatchedImageDataset(root_dir=LOCAL_PATCH_CACHE_DIR, transform=transform)
        if len(dataset) == 0:
            print(f"Error: No patches found in local cache '{LOCAL_PATCH_CACHE_DIR}'.")
            return
    except FileNotFoundError:
        print(f"Error: The local cache directory '{LOCAL_PATCH_CACHE_DIR}' was not found.")
        return

    # Standard DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=DATALOADER_WORKERS, 
        drop_last=True,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # Create model: ResNet50 backbone + projection head
    backbone = resnet50(weights=None)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    
    # Simple projection head
    projection_head = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    )
    
    backbone = backbone.to(DEVICE)
    projection_head = projection_head.to(DEVICE)
    
    # Combine parameters for optimizer
    params = list(backbone.parameters()) + list(projection_head.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    
    loss_fn = ContrastiveLoss(temperature=0.5)

    print(f"Starting training on {len(dataset)} image patches for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        backbone.train()
        projection_head.train()
        
        for view1, view2 in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            
            # Forward pass through backbone and projection head
            features1 = backbone(view1)
            features2 = backbone(view2)
            
            z1 = projection_head(features1)
            z2 = projection_head(features2)
            
            # Compute loss
            loss = loss_fn(z1, z2)
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    print(f"Saving final model to local directory: {LOCAL_MODEL_OUTPUT_DIR}")
    os.makedirs(LOCAL_MODEL_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, "resnet50_backbone.pth")
    torch.save(backbone.state_dict(), save_path)
    
    print(f"✅ Training complete. Backbone saved to persistent storage at: {save_path}")

if __name__ == '__main__':
    # Use 'spawn' for better compatibility across platforms
    multiprocessing.set_start_method('spawn', force=True)
    main()
