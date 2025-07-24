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
        
        total_patches = 0
        for i, blob_name in enumerate(blobs):
            try:
                print(f"\n🖼️  Processing image {i+1}/{len(blobs)}: {blob_name}")
                
                # Download image from Azure
                blob_client = container_client.get_blob_client(blob_name)
                image_bytes = blob_client.download_blob().readall()
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                width, height = img.size
                print(f"   Image size: {width}x{height}")
                
                # Calculate number of patches
                patches_x = (width - patch_size) // patch_size + 1
                patches_y = (height - patch_size) // patch_size + 1
                patches_per_image = patches_x * patches_y
                print(f"   Extracting {patches_per_image} patches ({patches_x}x{patches_y} grid)")
                
                # Extract 224x224 patches
                patch_count = 0
                for y in range(0, height - patch_size + 1, patch_size):
                    for x in range(0, width - patch_size + 1, patch_size):
                        patch = img.crop((x, y, x + patch_size, y + patch_size))
                        self.patches.append(patch)
                        patch_count += 1
                
                total_patches += patch_count
                print(f"   ✅ Successfully extracted {patch_count} patches")
                        
            except Exception as e:
                print(f"   ❌ Error processing {blob_name}: {e}")
        
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

def main():
    # Azure connection string
    connection_string = "DefaultEndpointsProtocol=https;AccountName=waypointtransit;AccountKey=65dhRHnC5SzoXBuf7XkiooDgoIinVGvwn4C3SZ8vkiH1Duqz6UMXT7ASuWSIGlRsDLZ7BOyt20cp+AStwkMVkg==;EndpointSuffix=core.windows.net"
    
    # Initialize wandb (will prompt for login on first run)
    wandb.init(project="byol-simple", config={
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 1e-3,
        "patch_size": 224,
        "ema_decay": 0.99,
        "azure_container": "resnet",
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1
    })
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple augmentations for BYOL
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
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
    print("\n🚀 Starting BYOL Training Pipeline")
    print("="*60)
    print("Creating patches from Azure images...")
    dataset = PatchDataset(
        azure_container=config.azure_container,
        connection_string=connection_string,
        patch_size=config.patch_size, 
        transform=TwoViewTransform(transform)
    )
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total patches available: {len(dataset)}")
    
    # Split dataset into train/val/test
    print("\n🔀 Splitting dataset...")
    total_size = len(dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"   Calculating splits:")
    print(f"   - {config.train_split*100:.0f}% train = {train_size} patches")
    print(f"   - {config.val_split*100:.0f}% val = {val_size} patches")
    print(f"   - {config.test_split*100:.0f}% test = {test_size} patches")
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n✅ Dataset split complete!")
    print(f"   Train: {len(train_dataset)} patches")
    print(f"   Val: {len(val_dataset)} patches")
    print(f"   Test: {len(test_dataset)} patches")
    print("="*60)
    wandb.log({
        "total_patches": total_size,
        "train_patches": len(train_dataset),
        "val_patches": len(val_dataset),
        "test_patches": len(test_dataset)
    })
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"\n🔧 Initializing model on {device.upper()}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Epochs: {config.epochs}")
    
    # Create BYOL model
    backbone = resnet18(pretrained=False)
    backbone.fc = nn.Identity()
    
    online_network = nn.Sequential(
        backbone,
        BYOLProjectionHead(512, 4096, 256)
    ).to(device)
    
    target_network = copy.deepcopy(online_network).to(device)
    prediction_head = BYOLPredictionHead(256, 4096, 256).to(device)
    
    print("   ✅ Model initialized successfully!")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(
        list(online_network.parameters()) + list(prediction_head.parameters()), 
        lr=config.learning_rate
    )
    criterion = NegativeCosineSimilarity()
    
    # Watch model
    wandb.watch(online_network, criterion, log="all")
    print("   ✅ Connected to Weights & Biases")
    print("="*60)
    
    # Training loop
    print("\n🏃 Starting training...")
    best_val_loss = float('inf')
    
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
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
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
        
        wandb.log({
            "epoch": epoch + 1, 
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'backbone': online_network[0].state_dict(),
                'full_model': online_network.state_dict(),
                'epoch': epoch
            }, "best_model.pth")
            print(f"   🏆 New best model saved! (Val Loss: {avg_val_loss:.4f})")
        else:
            print(f"   Current best Val Loss: {best_val_loss:.4f}")
    
    
    print("\n="*60)
    print("✅ Training completed!")
    
    # Save final model and log as artifact
    print("\n💾 Saving models...")
    torch.save(online_network[0].state_dict(), "final_backbone.pth")
    print("   Saved final_backbone.pth")
    
    artifact = wandb.Artifact("byol-model", type="model")
    artifact.add_file("best_model.pth")
    artifact.add_file("final_backbone.pth")
    wandb.log_artifact(artifact)
    print("   ✅ Models uploaded to W&B")
    
    # Test evaluation on best model
    print("\n🧪 Evaluating on test set...")
    checkpoint = torch.load("best_model.pth")
    online_network.load_state_dict(checkpoint['full_model'])
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
    print("="*60)
    
    wandb.finish()
    print("\n👋 All done! Check your results at https://wandb.ai")

if __name__ == "__main__":
    main()
