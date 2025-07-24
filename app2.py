import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
import copy
import wandb
from PIL import Image
import os
import glob

class PatchDataset(Dataset):
    def __init__(self, image_dir, patch_size=224, transform=None):
        self.transform = transform
        self.patches = []
        
        # Process all images in directory
        for img_path in glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True) + \
                       glob.glob(os.path.join(image_dir, "**/*.png"), recursive=True):
            try:
                img = Image.open(img_path).convert('RGB')
                width, height = img.size
                
                # Extract 224x224 patches
                for y in range(0, height - patch_size + 1, patch_size):
                    for x in range(0, width - patch_size + 1, patch_size):
                        patch = img.crop((x, y, x + patch_size, y + patch_size))
                        self.patches.append(patch)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
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
    # Initialize wandb
    wandb.init(project="byol-simple", config={
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 1e-3,
        "patch_size": 224,
        "ema_decay": 0.99,
        "data_dir": "resnet-data"
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
    print("Creating patches from images...")
    dataset = PatchDataset(config.data_dir, patch_size=config.patch_size, 
                          transform=TwoViewTransform(transform))
    print(f"Total patches: {len(dataset)}")
    wandb.log({"total_patches": len(dataset)})
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True)
    
    # Create BYOL model
    backbone = resnet18(pretrained=False)
    backbone.fc = nn.Identity()
    
    online_network = nn.Sequential(
        backbone,
        BYOLProjectionHead(512, 4096, 256)
    ).to(device)
    
    target_network = copy.deepcopy(online_network).to(device)
    prediction_head = BYOLPredictionHead(256, 4096, 256).to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(
        list(online_network.parameters()) + list(prediction_head.parameters()), 
        lr=config.learning_rate
    )
    criterion = NegativeCosineSimilarity()
    
    # Watch model
    wandb.watch(online_network, criterion, log="all")
    
    # Training loop
    for epoch in range(config.epochs):
        total_loss = 0
        for batch_idx, ((view1, view2), _) in enumerate(dataloader):
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
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                wandb.log({"batch_loss": loss.item()})
        
        # Log epoch metrics
        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")
    
    # Save final model
    torch.save(online_network[0].state_dict(), "backbone.pth")
    
    # Log model as artifact
    artifact = wandb.Artifact("byol-backbone", type="model")
    artifact.add_file("backbone.pth")
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    main()
