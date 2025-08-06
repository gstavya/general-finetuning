import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import resnet18
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
import copy
import wandb
from PIL import Image
import os
import glob
import io
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import numpy as np
import random

# Set seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_blob(args):
    blob_name, azure_container, connection_string, patch_size = args
    patches = []
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(azure_container)
        blob_client = container_client.get_blob_client(blob_name)
        image_bytes = blob_client.download_blob().readall()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = img.size
        
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                patches.append(patch)
        
        return blob_name, patches, None
    except Exception as e:
        return blob_name, [], str(e)

class PatchDataset(Dataset):
    def __init__(self, azure_container, connection_string, patch_size=224, transform=None):
        self.transform = transform
        self.patches = []
        
        # Connect to Azure
        print(f"\n🔌 Connecting to Azure Storage...")
        print(f"   Container: {azure_container}")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(azure_container)
        
        # Process all images from Azure
        print("\n📥 Listing blobs in container...")
        blobs = [b.name for b in container_client.list_blobs() 
                if b.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   Found {len(blobs)} images to process")
        print(f"   Using {multiprocessing.cpu_count()} CPU cores for parallel processing")
        
        # Process images in parallel
        args_list = [(blob_name, azure_container, connection_string, patch_size) for blob_name in blobs]
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(process_blob, args_list)
        
        total_patches = 0
        for blob_name, patches, error in results:
            if error:
                print(f"   ❌ Error processing {blob_name}: {error}")
            else:
                self.patches.extend(patches)
                total_patches += len(patches)
                print(f"   ✅ {blob_name}: {len(patches)} patches")
        
        print(f"\n📊 Total patches created: {total_patches}")
        print("="*60)
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.transform:
            return self.transform(patch), 0
        return patch, 0

def update_moving_average(ema_model, model, decay=0.99):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data

def upload_checkpoint_to_azure(connection_string, container_name, checkpoint_dict, blob_name):
    """Upload checkpoint dictionary directly to Azure blob storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Serialize checkpoint to bytes
        buffer = io.BytesIO()
        torch.save(checkpoint_dict, buffer)
        buffer.seek(0)
        
        # Upload to blob
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(buffer.getvalue(), overwrite=True)
        print(f"   ✅ Checkpoint uploaded to Azure: {blob_name}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to upload checkpoint to Azure: {e}")
        return False

def main():
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Azure connection string
    connection_string = "DefaultEndpointsProtocol=https;AccountName=waypointtransit;AccountKey=65dhRHnC5SzoXBuf7XkiooDgoIinVGvwn4C3SZ8vkiH1Duqz6UMXT7ASuWSIGlRsDLZ7BOyt20cp+AStwkMVkg==;EndpointSuffix=core.windows.net"
    
    # Initialize wandb with enhanced config
    wandb.init(project="byol-simple", config={
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 3e-4,
        "patch_size": 224,
        "ema_decay": 0.99,
        "azure_container": "resnet",
        "checkpoint_container": "resnet",  # Container for saving checkpoints
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "weight_decay": 1e-4,
        "patience": 10,
        "lr_scheduler_patience": 5,
        "lr_scheduler_factor": 0.5,
        # Anti-overfitting measures
        "dropout_rate": 0.2,
        "gradient_clip_norm": 1.0,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "cutmix_prob": 0.5
    })
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced augmentations for better generalization
    strong_augmentations = transforms.Compose([
        transforms.RandomResizedCrop(config.patch_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),  # Added
        transforms.RandomRotation(degrees=15),  # Added
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),  # Added
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),  # Added
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Lighter augmentations for validation
    val_augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create transform that returns two views
    class TwoViewTransform:
        def __init__(self, transform):
            self.transform = transform
        def __call__(self, x):
            return self.transform(x), self.transform(x)
    
    # Load dataset with patches
    print("\n🚀 Starting BYOL Training Pipeline with Anti-Overfitting Measures")
    print("="*60)
    print("Creating patches from Azure images...")
    
    # Create datasets with different transforms
    train_dataset_raw = PatchDataset(
        azure_container=config.azure_container,
        connection_string=connection_string,
        patch_size=config.patch_size, 
        transform=None  # Will apply transform after splitting
    )
    
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total patches available: {len(train_dataset_raw)}")
    
    # Split dataset into train/val/test
    print("\n🔀 Splitting dataset...")
    total_size = len(train_dataset_raw)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"   Calculating splits:")
    print(f"   - {config.train_split*100:.0f}% train = {train_size} patches")
    print(f"   - {config.val_split*100:.0f}% val = {val_size} patches")
    print(f"   - {config.test_split*100:.0f}% test = {test_size} patches")
    
    train_indices, val_indices, test_indices = random_split(
        range(len(train_dataset_raw)), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create datasets with appropriate transforms
    class TransformedSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            patch = self.dataset.patches[actual_idx]
            if self.transform:
                return self.transform(patch), 0
            return patch, 0
    
    train_dataset = TransformedSubset(train_dataset_raw, train_indices, TwoViewTransform(strong_augmentations))
    val_dataset = TransformedSubset(train_dataset_raw, val_indices, TwoViewTransform(val_augmentations))
    test_dataset = TransformedSubset(train_dataset_raw, test_indices, TwoViewTransform(val_augmentations))
    
    print(f"\n✅ Dataset split complete!")
    print(f"   Train: {len(train_dataset)} patches (strong augmentations)")
    print(f"   Val: {len(val_dataset)} patches (light augmentations)")
    print(f"   Test: {len(test_dataset)} patches (light augmentations)")
    print("="*60)
    
    wandb.log({
        "total_patches": total_size,
        "train_patches": len(train_dataset),
        "val_patches": len(val_dataset),
        "test_patches": len(test_dataset)
    })
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)  # drop_last for consistent batch norm
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"\n🔧 Initializing model on {str(device).upper()}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Dropout rate: {config.dropout_rate}")
    print(f"   Gradient clipping: {config.gradient_clip_norm}")
    
    # Create BYOL model with dropout
    backbone = resnet18(pretrained=False)
    backbone.fc = nn.Identity()
    
    # Add dropout to projection head
    class ProjectionHeadWithDropout(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
            super().__init__()
            self.projection = BYOLProjectionHead(input_dim, hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout_rate)
        
        def forward(self, x):
            x = self.projection(x)
            x = self.dropout(x)
            return x
    
    online_network = nn.Sequential(
        backbone,
        ProjectionHeadWithDropout(512, 4096, 256, config.dropout_rate)
    ).to(device)
    
    target_network = copy.deepcopy(online_network).to(device)
    
    # Add dropout to prediction head
    class PredictionHeadWithDropout(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
            super().__init__()
            self.prediction = BYOLPredictionHead(input_dim, hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout_rate)
        
        def forward(self, x):
            x = self.prediction(x)
            x = self.dropout(x)
            return x
    
    prediction_head = PredictionHeadWithDropout(256, 4096, 256, config.dropout_rate).to(device)
    
    print("   ✅ Model initialized with dropout layers!")
    
    # Setup optimizer with gradient clipping
    optimizer = torch.optim.AdamW(  # Using AdamW for better weight decay
        list(online_network.parameters()) + list(prediction_head.parameters()), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    criterion = NegativeCosineSimilarity()
    
    # Learning rate schedulers
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.lr_scheduler_factor, 
        patience=config.lr_scheduler_patience,
        verbose=True
    )
    
    # Cosine annealing as backup
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Watch model
    wandb.watch(online_network, criterion, log="all")
    print("   ✅ Connected to Weights & Biases")
    print("="*60)
    
    # Training loop
    print("\n🏃 Starting training with anti-overfitting measures...")
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(config.epochs):
        print(f"\n📈 Epoch {epoch+1}/{config.epochs}")
        print("-"*40)
        
        # Training phase
        online_network.train()
        prediction_head.train()
        total_train_loss = 0
        
        print(f"Training on {len(train_loader)} batches...")
        for batch_idx, ((view1, view2), _) in enumerate(train_loader):
            view1, view2 = view1.to(device), view2.to(device)
            
            # Forward pass
            z1_online = online_network(view1)
            z2_online = online_network(view2)
            
            with torch.no_grad():
                z1_target = target_network(view1)
                z2_target = target_network(view2)
            
            p1 = prediction_head(z1_online)
            p2 = prediction_head(z2_online)
            
            # Calculate loss
            loss = 0.5 * (criterion(p1, z2_target.detach()) + criterion(p2, z1_target.detach()))
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(online_network.parameters()) + list(prediction_head.parameters()), 
                config.gradient_clip_norm
            )
            
            optimizer.step()
            
            # Update target network
            update_moving_average(target_network, online_network, config.ema_decay)
            
            total_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                wandb.log({"batch_loss": loss.item()})
                print(f"   Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        # Validation phase
        print(f"\nValidating on {len(val_loader)} batches...")
        online_network.eval()
        prediction_head.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for (view1, view2), _ in val_loader:
                view1, view2 = view1.to(device), view2.to(device)
                
                z1_online = online_network(view1)
                z2_online = online_network(view2)
                z1_target = target_network(view1)
                z2_target = target_network(view2)
                p1 = prediction_head(z1_online)
                p2 = prediction_head(z2_online)
                
                val_loss = 0.5 * (criterion(p1, z2_target) + criterion(p2, z1_target))
                total_val_loss += val_loss.item()
        
        # Log epoch metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Step the learning rate scheduler
        scheduler.step(avg_val_loss)
        cosine_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({
            "epoch": epoch + 1, 
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": current_lr,
            "train_val_gap": avg_train_loss - avg_val_loss  # Monitor overfitting
        })
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   Train-Val Gap: {avg_train_loss - avg_val_loss:.4f}")
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint every epoch to Azure
        checkpoint = {
            'epoch': epoch,
            'online_network_state_dict': online_network.state_dict(),
            'target_network_state_dict': target_network.state_dict(),
            'prediction_head_state_dict': prediction_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'config': dict(config)
        }
        
        # Upload checkpoint to Azure
        checkpoint_name = f"checkpoint_epoch_{epoch+1}.pth"
        print(f"\n💾 Saving checkpoint to Azure container 'resnet'...")
        upload_checkpoint_to_azure(connection_string, "resnet", checkpoint, checkpoint_name)
        
        # Also save locally
        torch.save(checkpoint, f"local_checkpoint_epoch_{epoch+1}.pth")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            
            # Save best model checkpoint
            best_checkpoint = checkpoint.copy()
            best_checkpoint['is_best'] = True
            
            torch.save(best_checkpoint, "best_model.pth")
            upload_checkpoint_to_azure(connection_string, "resnet", best_checkpoint, "best_model_checkpoint.pth")
            
            print(f"   🏆 New best model saved! (Val Loss: {avg_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"   No improvement for {epochs_without_improvement} epochs (best: {best_val_loss:.4f})")
            
            # Early stopping check
            if epochs_without_improvement >= config.patience:
                print(f"\n⛔ Early stopping triggered! No improvement for {config.patience} epochs.")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                break
    
    print("\n="*60)
    print("✅ Training completed!")
    
    # Save final model
    print("\n💾 Saving final models...")
    final_checkpoint = {
        'epoch': epoch,
        'backbone_state_dict': online_network[0].state_dict(),
        'full_model_state_dict': online_network.state_dict(),
        'config': dict(config),
        'best_val_loss': best_val_loss
    }
    
    torch.save(online_network[0].state_dict(), "final_backbone.pth")
    upload_checkpoint_to_azure(connection_string, "resnet", final_checkpoint, "final_model_checkpoint.pth")
    print("   ✅ Final model uploaded to Azure")
    
    # Create and log artifact
    artifact = wandb.Artifact("byol-model", type="model")
    artifact.add_file("best_model.pth")
    artifact.add_file("final_backbone.pth")
    wandb.log_artifact(artifact)
    print("   ✅ Models uploaded to W&B")
    
    # Test evaluation on best model
    print("\n🧪 Evaluating on test set...")
    checkpoint = torch.load("best_model.pth")
    online_network.load_state_dict(checkpoint['online_network_state_dict'])
    target_network.load_state_dict(checkpoint['target_network_state_dict'])
    prediction_head.load_state_dict(checkpoint['prediction_head_state_dict'])
    print(f"   Loaded best model from epoch {checkpoint['epoch']+1}")
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    online_network.eval()
    prediction_head.eval()
    total_test_loss = 0
    
    print(f"   Testing on {len(test_loader)} batches...")
    with torch.no_grad():
        for (view1, view2), _ in test_loader:
            view1, view2 = view1.to(device), view2.to(device)
            
            z1_online = online_network(view1)
            z2_online = online_network(view2)
            z1_target = target_network(view1)
            z2_target = target_network(view2)
            p1 = prediction_head(z1_online)
            p2 = prediction_head(z2_online)
            
            test_loss = 0.5 * (criterion(p1, z2_target) + criterion(p2, z1_target))
            total_test_loss += test_loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    wandb.log({"test_loss": avg_test_loss})
    
    print(f"\n🏁 Final Results:")
    print(f"   Best Validation Loss: {best_val_loss:.4f}")
    print(f"   Test Loss: {avg_test_loss:.4f}")
    print(f"   Generalization Gap (Val-Test): {abs(best_val_loss - avg_test_loss):.4f}")
    print("="*60)
    
    wandb.finish()
    print("\n👋 All done! Check your results at https://wandb.ai")

if __name__ == "__main__":
    # Ensure fork method for multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        multiprocessing.set_start_method('fork', force=True)
    main()
