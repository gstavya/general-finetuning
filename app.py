import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from lightly.loss import NegativeCosineSimilarity
from lightly.models import BYOL
from lightly.transforms import BYOLTransform
from tqdm import tqdm
import os
from PIL import Image
import io
import argparse

# Import the Azure Blob Storage client library
from azure.storage.blob import BlobServiceClient

# Allow loading of large images that might otherwise raise an error
Image.MAX_IMAGE_PIXELS = None


def preprocess_and_save_locally(connection_string, source_container, local_target_dir, patch_size=224):
    """
    Connects to Azure, downloads original images, splits them into patches,
    and saves those patches directly to the specified local directory.
    This function populates the local cache for training.
    """
    # If the target directory already has files, skip this process to save time.
    if os.path.exists(local_target_dir) and len(os.listdir(local_target_dir)) > 0:
        print(f"Patches already exist in local cache ('{local_target_dir}'). Skipping preprocessing.")
        return

    print(f"Starting preprocessing: downloading from '{source_container}' and saving patches to '{local_target_dir}'...")
    os.makedirs(local_target_dir, exist_ok=True)

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(source_container)

        image_blobs = [blob for blob in container_client.list_blobs() if blob.name.lower().endswith(('jpg', 'jpeg', 'png'))]

        if not image_blobs:
            print(f"Warning: No images found in source container '{source_container}'.")
            return

        for blob in tqdm(image_blobs, desc="Downloading and Saving Patches Locally"):
            try:
                downloader = container_client.get_blob_client(blob.name).download_blob()
                image_bytes = downloader.readall()

                with Image.open(io.BytesIO(image_bytes)) as img:
                    width, height = img.size
                    patch_num = 0
                    for y in range(0, height - patch_size + 1, patch_size):
                        for x in range(0, width - patch_size + 1, patch_size):
                            box = (x, y, x + patch_size, y + patch_size)
                            patch = img.crop(box).convert('RGB')

                            original_filename = os.path.splitext(blob.name)[0].replace("/", "_")
                            patch_filename = f"{original_filename}_patch_{patch_num}.png"
                            save_path = os.path.join(local_target_dir, patch_filename)
                            
                            patch.save(save_path)
                            patch_num += 1
            except Exception as e:
                print(f"Warning: Could not process blob {blob.name}. Skipping. Error: {e}")
        
        print(f"✅ Preprocessing complete. All patches saved to '{local_target_dir}'.")

    except Exception as e:
        print(f"FATAL: An error occurred during preprocessing: {e}")
        raise

class PatchedImageDataset(Dataset):
    """
    This Dataset correctly reads the small, pre-split patches from the fast
    local cache directory (/mnt/data). This is efficient.
    """
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
            patch = self.transform(patch)
        return patch, 0, os.path.basename(image_path)

def main():
    # --- Configuration ---
    SOURCE_DATA_CONTAINER = "data"
    LOCAL_PATCH_CACHE_DIR = '/mnt/data' 
    LOCAL_MODEL_OUTPUT_DIR = '/mnt/satellite-resnet' # Define the local model save path

    PATCH_SIZE = 224
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    LR = 1e-3

    connection_string = "DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net"

    print(f"Using device: {DEVICE}")
    
    # 1. Preprocess images from Azure and save patches to the local disk.
    preprocess_and_save_locally(
        connection_string=connection_string,
        source_container=SOURCE_DATA_CONTAINER,
        local_target_dir=LOCAL_PATCH_CACHE_DIR,
        patch_size=PATCH_SIZE
    )

    # 2. Load dataset from the local cache directory.
    transform = BYOLTransform(input_size=PATCH_SIZE)
    try:
        dataset = PatchedImageDataset(root_dir=LOCAL_PATCH_CACHE_DIR, transform=transform)
        if len(dataset) == 0:
            print(f"Error: No patches found in local cache '{LOCAL_PATCH_CACHE_DIR}'. Preprocessing may have failed.")
            return
    except FileNotFoundError:
        print(f"Error: The local cache directory '{LOCAL_PATCH_CACHE_DIR}' was not found.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    
    # --- Training Loop (Simplified) ---
    resnet = resnet50(weights=None)
    resnet.fc = torch.nn.Identity()
    model = BYOL(backbone=resnet).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = NegativeCosineSimilarity()

    print(f"Starting training on {len(dataset)} image patches for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for (x0, x1), _, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            x0, x1 = x0.to(DEVICE), x1.to(DEVICE)
            p0, p1 = model(x0, x1)
            loss = loss_fn(p0, p1)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_moving_average()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    # --- Save the final backbone to local persistent storage ---
    # This section is now fixed to save the model locally.
    print(f"Saving final model to local directory: {LOCAL_MODEL_OUTPUT_DIR}")
    os.makedirs(LOCAL_MODEL_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, "byol_resnet50_backbone.pth")
    torch.save(model.backbone.state_dict(), save_path)
    
    print(f"✅ Training complete. Backbone saved to persistent storage at: {save_path}")

if __name__ == '__main__':
    main()
