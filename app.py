import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.v2 as T_v2
from torch.utils.data import DataLoader, Dataset
# --- MODIFICATION: Changed resnet50 to resnet18 ---
from torchvision.models import resnet18, ResNet18_Weights
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from tqdm import tqdm
import os
from PIL import Image
import io
from azure.storage.blob import BlobServiceClient
import multiprocessing
import copy
import glob # MODIFICATION: Added for finding checkpoint files

# Allow loading of large images that might otherwise raise an error
Image.MAX_IMAGE_PIXELS = None

# --- Worker function for parallel preprocessing (No changes needed) ---
def process_blob(args):
    """
    Worker function: downloads, splits, and saves patches for a single image blob.
    """
    blob_name, connection_string, source_container, local_target_dir, patch_size = args
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

# --- Preprocessing orchestrator (No changes needed) ---
def preprocess_and_save_locally(connection_string, source_container, local_target_dir, patch_size=224):
    if os.path.exists(local_target_dir) and len(os.listdir(local_target_dir)) > 0:
        print(f"Patches already exist in local cache ('{local_target_dir}'). Skipping preprocessing.")
        return
    print(f"Starting parallel preprocessing: downloading from '{source_container}'...")
    os.makedirs(local_target_dir, exist_ok=True)
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(source_container)
        image_blobs = [blob.name for blob in container_client.list_blobs() if blob.name.lower().endswith(('jpg', 'jpeg', 'png'))]
        if not image_blobs:
            print(f"Warning: No images found in source container '{source_container}'.")
            return
        tasks = [(blob_name, connection_string, source_container, local_target_dir, patch_size) for blob_name in image_blobs]
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(process_blob, tasks), total=len(tasks), desc="Processing Images in Parallel"):
                pass
        print(f"✅ Parallel preprocessing complete. All patches saved to '{local_target_dir}'.")
    except Exception as e:
        print(f"FATAL: An error occurred during preprocessing orchestration: {e}")
        raise

# --- Custom Dataset (No changes needed) ---
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
            view1, view2 = self.transform(patch)
            return (view1, view2), 0
        return patch, 0

# --- Manual Exponential Moving Average (EMA) update function (No changes needed) ---
def update_moving_average(ema_model, model, decay):
    """
    Updates the weights of the ema_model (target) to be a moving average
    of the model's (online) weights.
    """
    with torch.no_grad():
        for target_param, online_param in zip(ema_model.parameters(), model.parameters()):
            target_param.data = decay * target_param.data + (1 - decay) * online_param.data

def main():
    # --- Configuration ---
    SOURCE_DATA_CONTAINER = "data"
    LOCAL_PATCH_CACHE_DIR = '/mnt/data'
    # --- MODIFICATION: Consistent naming for output directory ---
    LOCAL_MODEL_OUTPUT_DIR = '/mnt/satellite-resnet2'
    PATCH_SIZE = 224
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATALOADER_WORKERS = 4 if torch.cuda.is_available() else 0
    LR = 1e-3
    EMA_DECAY = 0.999 # Standard decay rate for BYOL

    connection_string = "DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net"
    if not connection_string:
        raise ValueError("Azure Storage connection string is not set.")

    print(f"Using device: {DEVICE}")
    preprocess_and_save_locally(
        connection_string=connection_string,
        source_container=SOURCE_DATA_CONTAINER,
        local_target_dir=LOCAL_PATCH_CACHE_DIR,
        patch_size=PATCH_SIZE
    )

    # --- Data Transforms ---
    v2_transforms = T_v2.Compose([
        T_v2.ToImage(),
        T_v2.ToDtype(torch.float32, scale=True)
    ])
    
    class BYOLTransform:
        def __init__(self, size):
            self.transform = T.Compose([
                T.RandomResizedCrop(size=size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                v2_transforms,
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        def __call__(self, image):
            return self.transform(image), self.transform(image)
            
    transform = BYOLTransform(size=PATCH_SIZE)

    dataset = PatchedImageDataset(root_dir=LOCAL_PATCH_CACHE_DIR, transform=transform)
    if len(dataset) == 0:
        print(f"Error: No patches found in local cache '{LOCAL_PATCH_CACHE_DIR}'. Aborting.")
        return

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=DATALOADER_WORKERS, drop_last=True
    )
    print("Initializing default ResNet-18 backbone with pre-trained ImageNet weights...")
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity() 
    print("✅ ResNet-18 backbone initialized.")

    online_network = nn.Sequential(
        backbone, # Use the prepared resnet18 backbone
        BYOLProjectionHead(512, 4096, 256), # input_dim, hidden_dim, output_dim
    ).to(DEVICE)

    target_network = copy.deepcopy(online_network).to(DEVICE)
    prediction_head = BYOLPredictionHead(256, 4096, 256).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        list(online_network.parameters()) + list(prediction_head.parameters()), 
        lr=LR
    )
    loss_fn = NegativeCosineSimilarity()

    # --- MODIFICATION: Check for and load a checkpoint ---
    start_epoch = 0
    os.makedirs(LOCAL_MODEL_OUTPUT_DIR, exist_ok=True)
    checkpoint_files = sorted(glob.glob(os.path.join(LOCAL_MODEL_OUTPUT_DIR, "checkpoint_epoch_*.pth")))

    if checkpoint_files:
        latest_checkpoint_path = checkpoint_files[-1]
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
        
        online_network.load_state_dict(checkpoint['online_network_state_dict'])
        target_network.load_state_dict(checkpoint['target_network_state_dict'])
        prediction_head.load_state_dict(checkpoint['prediction_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"✅ Resumed from epoch {start_epoch}.")
    else:
        print("No checkpoint found. Starting training from scratch.")
    # --- END MODIFICATION ---

    # --- Training Loop ---
    # --- MODIFICATION: Adjust range to account for start_epoch ---
    print(f"Starting training on {len(dataset)} image patches from epoch {start_epoch + 1} to {NUM_EPOCHS}...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        total_loss = 0.0
        online_network.train()
        prediction_head.train()
        
        for (view1, view2), _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)

            # Forward pass through online network
            z0_online = online_network(view1)
            z1_online = online_network(view2)
            
            # Forward pass through target network (no gradients)
            with torch.no_grad():
                z0_target = target_network(view1)
                z1_target = target_network(view2)

            # Predictions
            p0 = prediction_head(z0_online)
            p1 = prediction_head(z1_online)

            # Calculate loss
            loss = 0.5 * (loss_fn(p0, z1_target.detach()) + loss_fn(p1, z0_target.detach()))
            total_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network using exponential moving average
            update_moving_average(target_network, online_network, decay=EMA_DECAY)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # --- MODIFICATION: Save checkpoint every 5 epochs ---
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}...")
            
            # The checkpoint includes everything needed to resume
            torch.save({
                'epoch': epoch,
                'online_network_state_dict': online_network.state_dict(),
                'target_network_state_dict': target_network.state_dict(),
                'prediction_head_state_dict': prediction_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print("✅ Checkpoint saved.")
        # --- END MODIFICATION ---


    # --- Save Final Model ---
    # *** MODIFICATION: Changed save path to reflect resnet18 architecture ***
    print(f"Saving final model backbone to local directory: {LOCAL_MODEL_OUTPUT_DIR}")
    os.makedirs(LOCAL_MODEL_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, "resnet18_backbone_final.pth")
    
    # The backbone is the first element of the online_network sequential model
    torch.save(online_network[0].state_dict(), save_path)
    print(f"✅ Training complete. Backbone saved to persistent storage at: {save_path}")

if __name__ == '__main__':
    # Set multiprocessing start method for compatibility if needed
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        multiprocessing.set_start_method('fork', force=True)
    main()
