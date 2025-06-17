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
import argparse # NEW: To handle command-line arguments (e.g., --preprocess)

# NEW: Import the Azure Blob Storage client library
from azure.storage.blob import BlobServiceClient

# Allow loading of large images that might otherwise raise an error
Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------------
# ✅ NEW: Function to preprocess images and UPLOAD patches to Azure
# --------------------------------------------------------------------------------
def preprocess_and_upload_patches(connection_string, source_container, dest_container, patch_size=224):
    """
    Connects to Azure, downloads images from a source container, splits them
    into patches, and uploads the patches to a destination container.
    """
    print(f"Starting preprocessing...")
    print(f"Source container: '{source_container}'")
    print(f"Destination container: '{dest_container}'")

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Client for the source container with original images
        source_container_client = blob_service_client.get_container_client(source_container)
        
        # Client for the destination container for patches
        dest_container_client = blob_service_client.get_container_client(dest_container)
        dest_container_client.create_container_if_not_exists() # Ensure dest container exists

        image_blobs = [blob for blob in source_container_client.list_blobs() if blob.name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

        if not image_blobs:
            print(f"Warning: No images found in source container '{source_container}'.")
            return

        for blob in tqdm(image_blobs, desc="Preprocessing and Uploading Patches"):
            try:
                downloader = source_container_client.get_blob_client(blob.name).download_blob()
                image_bytes = downloader.readall()

                with Image.open(io.BytesIO(image_bytes)) as img:
                    width, height = img.size
                    patch_num = 0
                    for y in range(0, height - patch_size + 1, patch_size):
                        for x in range(0, width - patch_size + 1, patch_size):
                            box = (x, y, x + patch_size, y + patch_size)
                            patch = img.crop(box).convert('RGB')

                            # Save patch to an in-memory buffer instead of a file
                            buffer = io.BytesIO()
                            patch.save(buffer, format='PNG')
                            buffer.seek(0) # Rewind the buffer to the beginning

                            # Define the name for the patch blob
                            original_filename = os.path.splitext(blob.name)[0].replace("/", "_")
                            patch_blob_name = f"{original_filename}_patch_{patch_num}.png"
                            
                            # Upload the patch from the buffer
                            dest_container_client.upload_blob(name=patch_blob_name, data=buffer, overwrite=True)
                            patch_num += 1
            except Exception as e:
                print(f"Warning: Could not process blob {blob.name}. Skipping. Error: {e}")
        
        print(f"✅ Preprocessing complete. All patches uploaded to container '{dest_container}'.")

    except Exception as e:
        print(f"FATAL: An error occurred during preprocessing: {e}")
        raise

# --------------------------------------------------------------------------------
# ✅ NEW: Function to download pre-split patches for training
# --------------------------------------------------------------------------------
def download_patches_for_training(connection_string, patch_container_name, local_dir):
    """
    Downloads all patches from a specified container to a local directory
    to be used for training.
    """
    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        print(f"Patches already exist in local cache ('{local_dir}'). Skipping download.")
        return

    print(f"Downloading patches from container '{patch_container_name}' to local cache '{local_dir}'...")
    os.makedirs(local_dir, exist_ok=True)

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(patch_container_name)
        
        blob_list = container_client.list_blobs()
        for blob in tqdm(blob_list, desc="Downloading Patches for Training"):
            local_path = os.path.join(local_dir, blob.name)
            with open(local_path, "wb") as download_file:
                download_file.write(container_client.get_blob_client(blob).download_blob().readall())
        
        print("✅ All patches downloaded successfully.")

    except Exception as e:
        print(f"FATAL: Could not download patches for training. Error: {e}")
        raise

# --------------------------------------------------------------------------------
# ✅ NEW: Function to upload the final model
# --------------------------------------------------------------------------------
def upload_model(connection_string, model_container_name, model_state_dict, model_filename):
    """
    Saves a model state dict to an in-memory buffer and uploads it to Azure.
    """
    print(f"Uploading model '{model_filename}' to container '{model_container_name}'...")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(model_container_name)
        container_client.create_container_if_not_exists()

        # Save the model state dict to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(model_state_dict, buffer)
        buffer.seek(0) # Rewind buffer

        # Upload the buffer's content
        container_client.upload_blob(name=model_filename, data=buffer, overwrite=True)
        print(f"✅ Model successfully uploaded to Azure Blob Storage.")

    except Exception as e:
        print(f"FATAL: Could not upload model. Error: {e}")
        raise

# --------------------------------------------------------------------------------
# This Dataset class remains unchanged, as it correctly reads from a local directory.
# --------------------------------------------------------------------------------
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
            patch = self.transform(patch)
        return patch, 0, os.path.basename(image_path)

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="ResNet Training with Azure Blob Storage")
    parser.add_argument('--preprocess', action='store_true', help="Run in preprocessing mode to split images and upload patches.")
    args = parser.parse_args()

    # --- Configuration ---
    SOURCE_DATA_CONTAINER = "data"
    SPLIT_DATA_CONTAINER = "data-split"
    MODEL_CONTAINER = "resnet-model"
    
    LOCAL_PATCH_CACHE_DIR = '/mnt/data'
    
    PATCH_SIZE = 224
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    LR = 1e-3

    # --- Securely get the connection string ---
    connection_string = "DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net"

    # --- Workflow Selection ---
    if args.preprocess:
        preprocess_and_upload_patches(
            connection_string=connection_string,
            source_container=SOURCE_DATA_CONTAINER,
            dest_container=SPLIT_DATA_CONTAINER,
            patch_size=PATCH_SIZE
        )
        return # Exit after preprocessing

    # --- Default Mode: Training ---
    print(f"Using device: {DEVICE}")
    
    # 1. Download patches from Azure to local cache for fast training
    download_patches_for_training(
        connection_string=connection_string,
        patch_container_name=SPLIT_DATA_CONTAINER,
        local_dir=LOCAL_PATCH_CACHE_DIR
    )

    # 2. Load dataset from the local cache
    transform = BYOLTransform(input_size=PATCH_SIZE)
    try:
        dataset = PatchedImageDataset(root_dir=LOCAL_PATCH_CACHE_DIR, transform=transform)
        if len(dataset) == 0:
            print(f"Error: No patches found in local cache '{LOCAL_PATCH_CACHE_DIR}'.")
            return
    except FileNotFoundError:
        print(f"Error: The local cache directory '{LOCAL_PATCH_CACHE_DIR}' was not found.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    # --- Training Loop (identical) ---
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

    # --- Upload the final model ---
    upload_model(
        connection_string=connection_string,
        model_container_name=MODEL_CONTAINER,
        model_state_dict=model.backbone.state_dict(),
        model_filename="byol_resnet50_backbone.pth"
    )

if __name__ == '__main__':
    main()
