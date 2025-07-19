import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.v2 as T_v2
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from tqdm import tqdm
import os
from PIL import Image
import io
from azure.storage.blob import BlobServiceClient
import multiprocessing
import copy
import glob
from sklearn.model_selection import train_test_split # MODIFICATION: To split the dataset

# Allow loading of large images that might otherwise raise an error
Image.MAX_IMAGE_PIXELS = None

# --- Worker function for parallel preprocessing (No changes) ---
def process_blob(args):
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

# --- Checkpoint upload function (No changes) ---
def upload_checkpoint_to_azure(connection_string, container_name, local_file_path, blob_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        try:
            container_client = blob_service_client.create_container(container_name)
        except Exception:
            container_client = blob_service_client.get_container_client(container_name)

        blob_client = container_client.get_blob_client(blob_name)
        print(f"Uploading '{blob_name}' to Azure container '{container_name}'...")
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print("✅ Upload complete.")
    except Exception as e:
        print(f"WARNING: Failed to upload to Azure. Error: {e}")

# --- Preprocessing orchestrator (No changes) ---
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

# --- MODIFICATION: Custom Dataset updated to handle different modes ---
class PatchedImageDataset(Dataset):
    def __init__(self, image_paths, transform=None, mode='train'):
        self.image_paths = image_paths
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        patch = Image.open(image_path).convert('RGB')

        if self.transform:
            if self.mode == 'train':
                view1, view2 = self.transform(patch)
                return (view1, view2), 0
            else: # For validation and test sets
                view = self.transform(patch)
                return view, 0

        return patch, 0

# --- Manual EMA update function (No changes) ---
def update_moving_average(ema_model, model, decay):
    with torch.no_grad():
        for target_param, online_param in zip(ema_model.parameters(), model.parameters()):
            target_param.data = decay * target_param.data + (1 - decay) * online_param.data

def main():
    # --- Configuration ---
    SOURCE_DATA_CONTAINER = "data"
    LOCAL_PATCH_CACHE_DIR = '/mnt/data'
    LOCAL_MODEL_OUTPUT_DIR = '/mnt/satellite-resnet2'
    PATCH_SIZE = 224
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATALOADER_WORKERS = 4 if torch.cuda.is_available() else 0
    LR = 1e-3
    EMA_DECAY = 0.999
    # MODIFICATION: Validation and Test set split percentage
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

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

    # --- MODIFICATION: Data Splitting (80% Train, 10% Val, 10% Test) ---
    all_patches = glob.glob(os.path.join(LOCAL_PATCH_CACHE_DIR, '*.png'))
    if not all_patches:
        print(f"Error: No patches found in '{LOCAL_PATCH_CACHE_DIR}'. Aborting.")
        return

    # First split: 80% train, 20% temp (for val + test)
    train_paths, temp_paths = train_test_split(
        all_patches, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=42
    )
    # Second split: Split the 20% temp into 10% val and 10% test
    val_paths, test_paths = train_test_split(
        temp_paths, test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)), random_state=42
    )

    print("-" * 50)
    print(f"Total patches: {len(all_patches)}")
    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")
    print(f"Test set size: {len(test_paths)}")
    print("-" * 50)

    # --- Data Transforms ---
    v2_transforms = T_v2.Compose([
        T_v2.ToImage(),
        T_v2.ToDtype(torch.float32, scale=True)
    ])

    # Transform for training (with heavy augmentation)
    train_transform = T.Compose([
        T.RandomResizedCrop(size=PATCH_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(23)], p=0.5),
        T.RandomSolarize(192.0, p=0.2),
        v2_transforms,
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Simple transform for validation and testing (no augmentation)
    val_test_transform = T.Compose([
        T.Resize((PATCH_SIZE, PATCH_SIZE)),
        v2_transforms,
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class BYOLTransformWrapper:
        def __init__(self, transform):
            self.transform = transform
        def __call__(self, image):
            return self.transform(image), self.transform(image)

    # --- MODIFICATION: Create Datasets and DataLoaders for all sets ---
    train_dataset = PatchedImageDataset(train_paths, transform=BYOLTransformWrapper(train_transform), mode='train')
    val_dataset = PatchedImageDataset(val_paths, transform=val_test_transform, mode='val')
    test_dataset = PatchedImageDataset(test_paths, transform=val_test_transform, mode='test')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=DATALOADER_WORKERS, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=DATALOADER_WORKERS, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=DATALOADER_WORKERS, drop_last=False
    )

    # --- Initialize Model Structure (No changes) ---
    print("Initializing ResNet-18 model structure...")
    backbone = resnet18(weights=IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    online_network = nn.Sequential(
        backbone,
        BYOLProjectionHead(512, 4096, 256),
    ).to(DEVICE)
    target_network = copy.deepcopy(online_network).to(DEVICE)
    prediction_head = BYOLPredictionHead(256, 4096, 256).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(online_network.parameters()) + list(prediction_head.parameters()), lr=LR
    )
    loss_fn = NegativeCosineSimilarity()

    # --- Checkpoint Loading (No changes, but now tracks best_val_loss) ---
    start_epoch = 0
    best_val_loss = float('inf')
    os.makedirs(LOCAL_MODEL_OUTPUT_DIR, exist_ok=True)
    CHECKPOINT_CONTAINER = "resnet18"
    CHECKPOINT_BLOB_NAME = "checkpoint_epoch_96.pth"
    print(f"Attempting to load checkpoint '{CHECKPOINT_BLOB_NAME}'...")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=CHECKPOINT_CONTAINER, blob=CHECKPOINT_BLOB_NAME)
        downloader = blob_client.download_blob()
        buffer = io.BytesIO(downloader.readall())
        checkpoint = torch.load(buffer, map_location=DEVICE)

        online_network.load_state_dict(checkpoint['online_network_state_dict'])
        target_network.load_state_dict(checkpoint['target_network_state_dict'])
        prediction_head.load_state_dict(checkpoint['prediction_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # If best_val_loss was saved in checkpoint, load it. Otherwise, keep it as infinity.
        best_val_loss = checkpoint.get('best_val_loss', float('inf')) 

        print(f"✅ State loaded. Resuming from epoch {start_epoch} with best validation loss {best_val_loss:.4f}.")
    except Exception as e:
        print(f"INFO: Could not load checkpoint from Azure. Starting from scratch. Error: {e}")

    # --- Training & Validation Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        # --- Training Phase ---
        online_network.train()
        prediction_head.train()
        total_train_loss = 0.0
        for (view1, view2), _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            optimizer.zero_grad()

            z0_online, z1_online = online_network(view1), online_network(view2)
            with torch.no_grad():
                z0_target, z1_target = target_network(view1), target_network(view2)
            p0, p1 = prediction_head(z0_online), prediction_head(z1_online)

            loss = 0.5 * (loss_fn(p0, z1_target.detach()) + loss_fn(p1, z0_target.detach()))
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            update_moving_average(target_network, online_network, decay=EMA_DECAY)

        avg_train_loss = total_train_loss / len(train_loader)

        # --- MODIFICATION: Validation Phase ---
        online_network.eval()
        prediction_head.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for view1, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  "):
                view1 = view1.to(DEVICE)
                view2 = view1.clone() # Create a second view for loss calculation

                z0_online, z1_online = online_network(view1), online_network(view2)
                z0_target, z1_target = target_network(view1), target_network(view2)
                p0, p1 = prediction_head(z0_online), prediction_head(z1_online)

                val_loss = 0.5 * (loss_fn(p0, z1_target.detach()) + loss_fn(p1, z0_target.detach()))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- MODIFICATION: Checkpoint Saving Logic ---
        # Save the latest checkpoint
        latest_checkpoint_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, "latest_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'online_network_state_dict': online_network.state_dict(),
            'target_network_state_dict': target_network.state_dict(),
            'prediction_head_state_dict': prediction_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss
        }, latest_checkpoint_path)

        # If current validation loss is the best, save a separate 'best' checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, "best_model_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'online_network_state_dict': online_network.state_dict(),
                'target_network_state_dict': target_network.state_dict(),
                'prediction_head_state_dict': prediction_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_checkpoint_path)
            print(f"🎉 New best model found! Val Loss: {avg_val_loss:.4f}. Saved to {best_checkpoint_path}")
            # Optionally upload the best model checkpoint to Azure
            upload_checkpoint_to_azure(connection_string, "resnet18", best_checkpoint_path, "best_model_checkpoint.pth")

    # --- MODIFICATION: Final Test Phase ---
    print("\n" + "="*50)
    print("Training finished. Evaluating best model on the test set...")
    best_model_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, "best_model_checkpoint.pth")
    try:
        # Load the best performing model
        best_checkpoint = torch.load(best_model_path)
        online_network.load_state_dict(best_checkpoint['online_network_state_dict'])
        target_network.load_state_dict(best_checkpoint['target_network_state_dict'])
        prediction_head.load_state_dict(best_checkpoint['prediction_head_state_dict'])
        print(f"✅ Successfully loaded best model from epoch {best_checkpoint['epoch']+1}.")

        online_network.eval()
        prediction_head.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for view1, _ in tqdm(test_loader, desc="[Final Test] "):
                view1 = view1.to(DEVICE)
                view2 = view1.clone() # Create a second view

                z0_online, z1_online = online_network(view1), online_network(view2)
                z0_target, z1_target = target_network(view1), target_network(view2)
                p0, p1 = prediction_head(z0_online), prediction_head(z1_online)

                test_loss = 0.5 * (loss_fn(p0, z1_target.detach()) + loss_fn(p1, z0_target.detach()))
                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"\n🏆 Final Test Loss (using best model): {avg_test_loss:.4f}")
        print("="*50)

    except FileNotFoundError:
        print("Error: `best_model_checkpoint.pth` not found. Could not run final test evaluation.")
    except Exception as e:
        print(f"An error occurred during final test evaluation: {e}")

    # --- Save Final Backbone ---
    print(f"Saving final model backbone from the last epoch to local directory...")
    final_backbone_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, "resnet18_backbone_final_epoch.pth")
    torch.save(online_network[0].state_dict(), final_backbone_path)
    print(f"✅ Final backbone saved to: {final_backbone_path}")


if __name__ == '__main__':
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        multiprocessing.set_start_method('fork', force=True)
    main()
