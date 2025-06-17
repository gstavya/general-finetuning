import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.loss import NegativeCosineSimilarity
from lightly.models import BYOL
from lightly.transforms import BYOLTransform
from tqdm import tqdm
import os
from PIL import Image
import io  # NEW: Required for reading image bytes from memory

# NEW: Import the Azure Blob Storage client library
from azure.storage.blob import BlobServiceClient

# Allow loading of large images that might otherwise raise an error
Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------------
# ✅ MODIFIED: Function to download and preprocess from Azure Blob Storage
# --------------------------------------------------------------------------------
def download_and_preprocess_from_azure(connection_string, container_name, target_dir, patch_size=224):
    """
    Connects to Azure Blob Storage, downloads images, splits them into patches,
    and saves them to a local target directory.

    Args:
        connection_string (str): The connection string for the Azure Storage Account.
        container_name (str): The name of the container holding the images.
        target_dir (str): The local directory where pre-split patches will be saved.
        patch_size (int): The height and width of the square patches.
    """
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"Patches already exist in persistent storage ('{target_dir}'). Skipping preprocessing.")
        return

    print(f"Connecting to Azure Blob Storage container '{container_name}'...")
    print(f"Downloading images and saving patches to persistent storage ('{target_dir}')...")
    os.makedirs(target_dir, exist_ok=True)

    try:
        # Initialize the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        blob_list = container_client.list_blobs()
        
        # Filter for common image file extensions
        image_blobs = [blob for blob in blob_list if blob.name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

        if not image_blobs:
            print(f"Warning: No images found in the Azure container '{container_name}'.")
            return

        for blob in tqdm(image_blobs, desc="Downloading & Saving Patches"):
            try:
                # Download blob content into memory
                blob_client = container_client.get_blob_client(blob.name)
                downloader = blob_client.download_blob()
                image_bytes = downloader.readall()

                # Open the image from the in-memory bytes
                with Image.open(io.BytesIO(image_bytes)) as img:
                    width, height = img.size
                    patch_num = 0
                    for y in range(0, height - patch_size + 1, patch_size):
                        for x in range(0, width - patch_size + 1, patch_size):
                            box = (x, y, x + patch_size, y + patch_size)
                            patch = img.crop(box)

                            original_filename = os.path.splitext(blob.name)[0].replace("/", "_")
                            patch_filename = f"{original_filename}_patch_{patch_num}.png"
                            save_path = os.path.join(target_dir, patch_filename)
                            
                            patch.convert('RGB').save(save_path)
                            patch_num += 1
            except Exception as e:
                print(f"Warning: Could not process blob {blob.name}. Skipping. Error: {e}")

    except Exception as e:
        print(f"FATAL: An error occurred while connecting to or reading from Azure: {e}")
        print("Please ensure your AZURE_STORAGE_CONNECTION_STRING is set correctly.")
        raise

# --------------------------------------------------------------------------------
# ✅ Dataset for Pre-split Patches (No changes needed here)
# --------------------------------------------------------------------------------
class PatchedImageDataset(Dataset):
    """
    A PyTorch Dataset that loads pre-split image patches directly from a directory.
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
    # ✅ MODIFIED: Azure Storage Configuration
    AZURE_ACCOUNT_NAME = "resnettrainingdata"
    AZURE_CONTAINER_NAME = "data"
    
    # This is the persistent storage provided by your ML platform
    PERSISTENT_PATCH_DIR = '/mnt/data' 
    
    PATCH_SIZE = 224
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    LR = 1e-3

    print(f"Using device: {DEVICE}")
    print(f"Sourcing images from Azure Blob Storage: '{AZURE_ACCOUNT_NAME}/{AZURE_CONTAINER_NAME}'")
    print(f"Persistent Patch Storage: '{PERSISTENT_PATCH_DIR}'")

    # ✅ MODIFIED: Get connection string from environment variable for security
    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        print("FATAL: Environment variable 'AZURE_STORAGE_CONNECTION_STRING' not set.")
        print("Please set this secret in your ML platform's environment settings.")
        return

    # ✅ MODIFIED: Run Preprocessing Step from Azure
    # This reads from Azure and saves to the persistent /mnt/data folder.
    # It will only run the first time.
    download_and_preprocess_from_azure(
        connection_string=DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net,
        container_name=AZURE_CONTAINER_NAME,
        target_dir=PERSISTENT_PATCH_DIR,
        patch_size=PATCH_SIZE
    )

    # ✅ 1. Load and augment dataset FROM THE PERSISTENT STORAGE directory
    transform = BYOLTransform(input_size=PATCH_SIZE)

    try:
        dataset = PatchedImageDataset(root_dir=PERSISTENT_PATCH_DIR, transform=transform)
        if len(dataset) == 0:
            print(f"Error: No preprocessed patches found in '{PERSISTENT_PATCH_DIR}'. Preprocessing may have failed.")
            return
    except FileNotFoundError:
        print(f"Error: The persistent patch directory '{PERSISTENT_PATCH_DIR}' was not found.")
        print("This may be normal on the first run. Preprocessing will attempt to create it.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    if len(dataloader) == 0:
        print("Error: Your DataLoader is empty. This is likely because the number of patches is smaller than the batch size.")
        return

    # --- (The rest of the training loop is identical) ---

    # 2. Define backbone + BYOL wrapper
    resnet = resnet50(weights=None)
    resnet.fc = torch.nn.Identity()
    model = BYOL(backbone=resnet).to(DEVICE)

    # 3. Optimizer + loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = NegativeCosineSimilarity()

    # 4. Training loop
    print(f"Starting training on {len(dataset)} image patches for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for (x0, x1), _, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            x0, x1 = x0.to(DEVICE), x1.to(DEVICE)
            predictions, projections = model(x0, x1)
            p0, p1 = predictions
            z0, z1 = projections
            loss = 0.5 * (loss_fn(p0, z1.detach()) + loss_fn(p1, z0.detach()))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if len(dataloader) > 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    # 5. Save the final backbone
    OUTPUT_DIR = "/mnt/satellite-resnet"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "byol_resnet50_backbone_patched.pth")
    torch.save(model.backbone.state_dict(), save_path)
    
    print(f"✅ Training complete. Backbone saved to persistent storage at: {save_path}")

if __name__ == '__main__':
    main()
