import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.v2 as T_v2
from torch.utils.data import DataLoader, Dataset
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
import glob
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import shutil # Import shutil for copying files

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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

# --- Checkpoint upload function (Modified for rank 0 only and epoch-based naming) ---
def upload_checkpoint_to_azure(connection_string, container_name, local_file_path, blob_name, rank):
    if rank != 0:  # Only rank 0 uploads
        return

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        try:
            container_client = blob_service_client.create_container(container_name)
        except Exception: # Container might already exist, so just get it
            container_client = blob_service_client.get_container_client(container_name)

        blob_client = container_client.get_blob_client(blob_name)
        print(f"Uploading checkpoint '{blob_name}' to Azure container '{container_name}'...")
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True) # Always overwrite
        print("✅ Upload complete.")
    except Exception as e:
        print(f"WARNING: Failed to upload '{blob_name}' to Azure. Error: {e}")

# --- Preprocessing orchestrator (Modified for rank 0 only) ---
def preprocess_and_save_locally(connection_string, source_container, local_target_dir, patch_size=224, rank=0):
    if rank != 0:  # Only rank 0 does preprocessing
        return
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

# --- Dataset Class (No changes) ---
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
            else:
                view = self.transform(patch)
                return view, 0
        return patch, 0

# --- GLOBAL SCOPE: BYOLTransformWrapper moved here to be picklable ---
class BYOLTransformWrapper:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, image):
        return self.transform(image), self.transform(image)

# --- Helper function for EMA Cosine Scheduling ---
def get_ema_decay(epoch, total_epochs, start_decay=0.99, end_decay=1.0):
    """
    Calculates the EMA decay for the current epoch using a cosine schedule.
    It gradually increases the decay from a starting value to an ending value.
    """
    cosine_schedule = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    return end_decay - (end_decay - start_decay) * cosine_schedule

# --- Manual EMA update function (No changes) ---
def update_moving_average(ema_model, model, decay):
    with torch.no_grad():
        for target_param, online_param in zip(ema_model.parameters(), model.parameters()):
            target_param.data = decay * target_param.data + (1 - decay) * online_param.data

# --- DDP Setup Functions ---
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    setup(rank, world_size)

    # --- Configuration ---
    SOURCE_DATA_CONTAINER = "data"
    LOCAL_PATCH_CACHE_DIR = 'data_patches'  # Changed to separate directory for patches
    LOCAL_MODEL_OUTPUT_DIR = 'satellite-resnet2'
    PATCH_SIZE = 224
    BATCH_SIZE = 512  # This is now the total batch size since we're using 1 GPU
    NUM_EPOCHS = 100
    DEVICE = f'cuda:{rank}'  # Each process uses its own GPU
    torch.cuda.set_device(rank)
    DATALOADER_WORKERS = 4
    LR = 1e-3 * world_size  # Scale learning rate with number of GPUs
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

    # Configuration for EMA scheduling
    START_EMA_DECAY = 0.96
    END_EMA_DECAY = 1.0

    # Azure Blob Storage connection string
    # WARNING: Hardcoding sensitive information like connection strings directly in code
    # is NOT recommended for production environments. Use environment variables,
    # Azure Key Vault, or similar secure methods.
    connection_string = "DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net"
    if not connection_string:
        raise ValueError("Azure Storage connection string is not set.")

    if rank == 0:
        print(f"Using {world_size} GPU(s) with DDP")
        print(f"Per-GPU batch size: {BATCH_SIZE}")
        print(f"Total effective batch size: {BATCH_SIZE * world_size}")
        print(f"Scaled learning rate: {LR}")

    # Only rank 0 does preprocessing
    preprocess_and_save_locally(
        connection_string=connection_string,
        source_container=SOURCE_DATA_CONTAINER,
        local_target_dir=LOCAL_PATCH_CACHE_DIR,
        patch_size=PATCH_SIZE,
        rank=rank
    )

    # Synchronize all processes after preprocessing
    dist.barrier()

    # --- Data Splitting (All ranks need same split) ---
    all_patches = glob.glob(os.path.join(LOCAL_PATCH_CACHE_DIR, '*.png'))
    if not all_patches:
        if rank == 0:
            print(f"Error: No patches found in '{LOCAL_PATCH_CACHE_DIR}'. Aborting.")
        cleanup()
        return

    # Use same random seed across all ranks for consistent splits
    train_paths, temp_paths = train_test_split(all_patches, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)), random_state=42)

    if rank == 0:
        print("-" * 50)
        print(f"Training set size: {len(train_paths)}, Validation set size: {len(val_paths)}, Test set size: {len(test_paths)}")
        print("-" * 50)

    # --- Data Transforms ---
    v2_transforms = T_v2.Compose([T_v2.ToImage(), T_v2.ToDtype(torch.float32, scale=True)])
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
    val_test_transform = T.Compose([
        T.Resize((PATCH_SIZE, PATCH_SIZE)), v2_transforms,
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Datasets and DataLoaders with DistributedSampler ---
    train_dataset = PatchedImageDataset(train_paths, transform=BYOLTransformWrapper(train_transform), mode='train')
    val_dataset = PatchedImageDataset(val_paths, transform=val_test_transform, mode='val')
    test_dataset = PatchedImageDataset(test_paths, transform=val_test_transform, mode='test')

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                             num_workers=DATALOADER_WORKERS, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
                            num_workers=DATALOADER_WORKERS, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler,
                             num_workers=DATALOADER_WORKERS, drop_last=False)

    # --- Model Initialization ---
    if rank == 0:
        print("Initializing ResNet-18 model structure...")

    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    online_network = nn.Sequential(backbone, BYOLProjectionHead(512, 4096, 256)).to(DEVICE)
    target_network = copy.deepcopy(online_network).to(DEVICE)
    prediction_head = BYOLPredictionHead(256, 4096, 256).to(DEVICE)

    # Wrap models with DDP
    online_network = DDP(online_network, device_ids=[rank])
    prediction_head = DDP(prediction_head, device_ids=[rank])
    # Note: target_network is not wrapped in DDP as it's updated via EMA

    optimizer = torch.optim.Adam(list(online_network.parameters()) + list(prediction_head.parameters()), lr=LR)
    loss_fn = NegativeCosineSimilarity()
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0)

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_loss = float('inf') # Initial value

    if rank == 0:
        os.makedirs(LOCAL_MODEL_OUTPUT_DIR, exist_ok=True)

    CHECKPOINT_CONTAINER = "resnet18-optimized"
    if rank == 0:
        print(f"Attempting to load the latest epoch checkpoint for resumption...")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(CHECKPOINT_CONTAINER)

        latest_blob_name = None
        max_epoch_found = -1
        for blob in container_client.list_blobs():
            if blob.name.startswith("model_checkpoint_epoch_") and blob.name.endswith(".pth"):
                try:
                    epoch_str = blob.name.split('_')[-1].split('.')[0]
                    epoch_num = int(epoch_str)
                    if epoch_num > max_epoch_found:
                        max_epoch_found = epoch_num
                        latest_blob_name = blob.name
                except ValueError:
                    continue

        if latest_blob_name:
            if rank == 0:
                print(f"Found latest checkpoint: '{latest_blob_name}'. Downloading...")
            blob_client = container_client.get_blob_client(latest_blob_name)
            downloader = blob_client.download_blob()
            buffer = io.BytesIO(downloader.readall())
            checkpoint = torch.load(buffer, map_location=DEVICE)

            online_network.module.load_state_dict(checkpoint['online_network_state_dict'])
            target_network.load_state_dict(checkpoint['target_network_state_dict'])
            prediction_head.module.load_state_dict(checkpoint['prediction_head_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            if rank == 0:
                print(f"✅ State loaded. Resuming from epoch {start_epoch}.")
        else:
            if rank == 0:
                print("INFO: No previous epoch checkpoints found. Starting from scratch.")
    except Exception as e:
        if rank == 0:
            print(f"INFO: An error occurred while trying to load latest checkpoint. Starting from scratch. Error: {e}")


    # --- Training & Validation Loop ---
    if rank == 0:
        print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        current_ema_decay = get_ema_decay(epoch, NUM_EPOCHS, START_EMA_DECAY, END_EMA_DECAY)

        # --- Training Phase ---
        online_network.train()
        prediction_head.train()
        total_train_loss = 0.0

        train_iter = train_loader
        if rank == 0:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        for (view1, view2), _ in train_iter:
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            optimizer.zero_grad()

            z0_online, z1_online = online_network(view1), online_network(view2)
            with torch.no_grad():
                z0_target, z1_target = target_network(view1), target_network(view2)
            p0, p1 = prediction_head(z0_online), prediction_head(z1_online)

            loss = 0.5 * (loss_fn(p0, z1_target.detach()) + loss_fn(p1, z0_target.detach()))
            total_train_loss += loss.item()

            loss.backward()
            update_moving_average(target_network, online_network.module, decay=current_ema_decay)
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_tensor = torch.tensor(avg_train_loss).to(DEVICE)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        avg_train_loss = train_loss_tensor.item()

        # --- Validation Phase ---
        online_network.eval()
        prediction_head.eval()
        total_val_loss = 0.0

        val_iter = val_loader
        if rank == 0:
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  ")

        with torch.no_grad():
            for view1, _ in val_iter:
                view1 = view1.to(DEVICE)
                view2 = view1.clone()
                z0_online, z1_online = online_network(view1), online_network(view2)
                z0_target, z1_target = target_network(view1), target_network(view2)
                p0, p1 = prediction_head(z0_online), prediction_head(z1_online)
                val_loss = 0.5 * (loss_fn(p0, z1_target.detach()) + loss_fn(p1, z0_target.detach()))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_tensor = torch.tensor(avg_val_loss).to(DEVICE)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        avg_val_loss = val_loss_tensor.item()

        # --- Epoch End: Step Scheduler and Log ---
        scheduler.step()
        if rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}, Current EMA Decay: {current_ema_decay:.6f}")

            # --- Checkpoint Saving for EVERY EPOCH (rank 0 only) ---
            current_epoch_checkpoint_filename = f"model_checkpoint_epoch_{epoch+1:03d}.pth"
            current_epoch_checkpoint_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, current_epoch_checkpoint_filename)

            torch.save({
                'epoch': epoch,
                'online_network_state_dict': online_network.module.state_dict(),
                'target_network_state_dict': target_network.state_dict(),
                'prediction_head_state_dict': prediction_head.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': min(best_val_loss, avg_val_loss)
            }, current_epoch_checkpoint_path)
            print(f"✅ Checkpoint for epoch {epoch+1} saved locally: {current_epoch_checkpoint_path}")

            upload_checkpoint_to_azure(
                connection_string,
                CHECKPOINT_CONTAINER,
                current_epoch_checkpoint_path,
                current_epoch_checkpoint_filename,
                rank
            )

            latest_resume_blob_name = "latest_resume_checkpoint.pth"
            latest_resume_checkpoint_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, latest_resume_blob_name)
            shutil.copyfile(current_epoch_checkpoint_path, latest_resume_checkpoint_path)
            upload_checkpoint_to_azure(
                connection_string,
                CHECKPOINT_CONTAINER,
                latest_resume_checkpoint_path,
                latest_resume_blob_name,
                rank
            )
            if rank == 0:
                print(f"✅ 'latest_resume_checkpoint.pth' updated in Azure.")

    # --- Final Test Phase ---
    if rank == 0:
        print("\n" + "="*50)
        print("Training finished. Evaluating model on the test set from the LAST completed epoch...")

    final_model_blob_name = "latest_resume_checkpoint.pth"
    final_model_path_local = os.path.join(LOCAL_MODEL_OUTPUT_DIR, final_model_blob_name)

    try:
        if not os.path.exists(final_model_path_local):
            if rank == 0:
                print(f"Downloading final checkpoint '{final_model_blob_name}' from Azure...")
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container=CHECKPOINT_CONTAINER, blob=final_model_blob_name)
            downloader = blob_client.download_blob()
            with open(final_model_path_local, "wb") as file:
                file.write(downloader.readall())
            if rank == 0:
                print("✅ Download complete.")

        final_checkpoint = torch.load(final_model_path_local, map_location=DEVICE)
        online_network.module.load_state_dict(final_checkpoint['online_network_state_dict'])
        target_network.load_state_dict(final_checkpoint['target_network_state_dict'])
        prediction_head.module.load_state_dict(final_checkpoint['prediction_head_state_dict'])

        if rank == 0:
            print(f"✅ Successfully loaded model from epoch {final_checkpoint['epoch']+1} for final test.")

        online_network.eval()
        prediction_head.eval()
        total_test_loss = 0.0

        test_iter = test_loader
        if rank == 0:
            test_iter = tqdm(test_loader, desc="[Final Test] ")

        with torch.no_grad():
            for view1, _ in test_iter:
                view1 = view1.to(DEVICE)
                view2 = view1.clone()
                z0_online, z1_online = online_network(view1), online_network(view2)
                z0_target, z1_target = target_network(view1), target_network(view2)
                p0, p1 = prediction_head(z0_online), prediction_head(z1_online)
                test_loss = 0.5 * (loss_fn(p0, z1_target.detach()) + loss_fn(p1, z0_target.detach()))
                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_tensor = torch.tensor(avg_test_loss).to(DEVICE)
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.AVG)
        avg_test_loss = test_loss_tensor.item()

        if rank == 0:
            print(f"\n🏆 Final Test Loss (using model from last epoch): {avg_test_loss:.4f}")
    except FileNotFoundError:
        if rank == 0:
            print(f"Error: Final checkpoint '{final_model_path_local}' not found. Could not run final test evaluation.")
    except Exception as e:
        if rank == 0:
            print(f"An error occurred during final test evaluation: {e}")

    # --- Save Final Backbone (rank 0 only) ---
    if rank == 0:
        print(f"Saving final model backbone from the last epoch to local directory...")
        final_backbone_path = os.path.join(LOCAL_MODEL_OUTPUT_DIR, "resnet18_backbone_final_epoch.pth")
        torch.save(online_network.module[0].state_dict(), final_backbone_path)
        print(f"✅ Final backbone saved to: {final_backbone_path}")

    cleanup()

def main():
    # Check how many GPUs are available
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPUs available. This script requires at least 1 GPU.")
    
    world_size = n_gpus  # Use all available GPUs
    print(f"Found {world_size} GPU(s). Using all of them for training.")
    
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
