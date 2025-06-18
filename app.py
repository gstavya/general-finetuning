import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from lightly.loss import NegativeCosineSimilarity
from lightly.models import BYOL
from lightly.data.collate import BYOLCollateFunction # Correct import for modern lightly
from tqdm import tqdm
import os
from PIL import Image
import io
import argparse
from azure.storage.blob import BlobServiceClient
import multiprocessing # NEW: Import for parallel processing

# Allow loading of large images that might otherwise raise an error
Image.MAX_IMAGE_PIXELS = None

# --- NEW: Worker function for parallel processing ---
# This function contains the logic to process a SINGLE blob. It will be
# executed by each worker in the multiprocessing pool.
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

# --- MODIFIED: The preprocessing function now orchestrates the parallel work ---
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

        # Create a pool of worker processes. os.cpu_count() uses all available cores.
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            # Use tqdm to show a progress bar for the parallel tasks
            for result in tqdm(pool.imap_unordered(process_blob, tasks), total=len(tasks), desc="Processing Images in Parallel"):
                # You can optionally print results or errors from workers here
                pass
        
        print(f"✅ Parallel preprocessing complete. All patches saved to '{local_target_dir}'.")

    except Exception as e:
        print(f"FATAL: An error occurred during preprocessing orchestration: {e}")
        raise

class PatchedImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        patch = Image.open(image_path).convert('RGB')
        return patch, 0, os.path.basename(image_path)

def main():
    SOURCE_DATA_CONTAINER = "data"
    LOCAL_PATCH_CACHE_DIR = '/mnt/data' 
    LOCAL_MODEL_OUTPUT_DIR = '/mnt/satellite-resnet'

    PATCH_SIZE = 224
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Recommended to set num_workers in DataLoader to 0 when using multiprocessing
    # heavily in the main script to avoid contention.
    DATALOADER_WORKERS = 4 if torch.cuda.is_available() else 0
    LR = 1e-3

    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        print("FATAL: Environment variable 'AZURE_STORAGE_CONNECTION_STRING' not set.")
        return

    print(f"Using device: {DEVICE}")
    
    preprocess_and_save_locally(
        connection_string=connection_string,
        source_container=SOURCE_DATA_CONTAINER,
        local_target_dir=LOCAL_PATCH_CACHE_DIR,
        patch_size=PATCH_SIZE
    )

    try:
        dataset = PatchedImageDataset(root_dir=LOCAL_PATCH_CACHE_DIR)
        if len(dataset) == 0:
            print(f"Error: No patches found in local cache '{LOCAL_PATCH_CACHE_DIR}'.")
            return
    except FileNotFoundError:
        print(f"Error: The local cache directory '{LOCAL_PATCH_CACHE_DIR}' was not found.")
        return

    # Use the BYOLCollateFunction for modern lightly API
    collate_fn = BYOLCollateFunction(input_size=PATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=DATALOADER_WORKERS, drop_last=True)
    
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

    print(f"Saving final model to local directory: {LOCAL_MODEL_OUTPUT_DIR}")
    os.makedirs(LOCAL_MODEL_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, "byol_resnet50_backbone.pth")
    torch.save(model.backbone.state_dict(), save_path)
    
    print(f"✅ Training complete. Backbone saved to persistent storage at: {save_path}")

if __name__ == '__main__':
    # This is important for multiprocessing to work correctly on all platforms
    multiprocessing.set_start_method('fork')
    main()
