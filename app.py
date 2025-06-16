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

# Allow loading of large images that might otherwise raise an error
Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------------
# ✅ Custom Dataset for Splitting Images into Patches (This part is correct)
# --------------------------------------------------------------------------------
class ImagePatchDataset(Dataset):
    """
    Custom PyTorch Dataset to split large images into smaller patches.

    This dataset scans a directory of images. For each image, it calculates
    all possible non-overlapping patches of a given size. The __getitem__ method
    then crops and returns a specific patch, to which data augmentations
    (like the BYOL transform) can be applied.
    """
    def __init__(self, root_dir, patch_size=224, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            patch_size (int): The height and width of the square patches.
            transform (callable, optional): Optional transform to be applied on a patch.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
        
        self.patches_map = []
        print("Pre-calculating image patches...")
        # Create a map of all possible patches from all images
        for image_path in tqdm(self.image_paths, desc="Scanning Images"):
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    # Calculate how many patches fit in the image
                    for y in range(0, height - self.patch_size + 1, self.patch_size):
                        for x in range(0, width - self.patch_size + 1, self.patch_size):
                            # Store the path and the top-left coordinate of the patch
                            self.patches_map.append((image_path, x, y))
            except Exception as e:
                print(f"Warning: Could not read or process {image_path}. Skipping. Error: {e}")

    def __len__(self):
        """Returns the total number of patches."""
        return len(self.patches_map)

    def __getitem__(self, idx):
        """
        Retrieves the patch at the given index, crops it, and applies transforms.
        """
        # Get the image path and patch coordinates from our map
        image_path, x, y = self.patches_map[idx]
        
        # Open the original large image
        original_image = Image.open(image_path).convert('RGB')
        
        # Define the crop box and extract the patch
        box = (x, y, x + self.patch_size, y + self.patch_size)
        patch = original_image.crop(box)
        
        # Apply the BYOL augmentations to the patch
        if self.transform:
            patch = self.transform(patch)

        # Return the transformed patch and dummy data to match dataloader expectations
        # The BYOL transform returns two augmented views (x0, x1) of the patch
        return patch, 0, os.path.basename(image_path)

def main():
    # --- Configuration ---
    DATA_DIR = 'data'
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

    # 1. Load and augment dataset using our new patch-based dataset
    # The default BYOLTransform is already set for an input size of 224
    transform = BYOLTransform()

    try:
        dataset = ImagePatchDataset(root_dir=DATA_DIR, patch_size=PATCH_SIZE, transform=transform)
        if len(dataset) == 0:
            print(f"Error: No images found or no patches could be created in the '{DATA_DIR}' directory.")
            print("Please ensure your images are larger than the patch size of 224x224.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{DATA_DIR}' was not found.")
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
        print("---")
        print("Error: Your DataLoader is empty.")
        print(f"This is likely because your total number of patches ({len(dataset)}) is smaller than your BATCH_SIZE ({BATCH_SIZE}).")
        print("\nPossible Solutions:")
        print("1. Add more/larger images to your dataset to generate more patches.")
        print(f"2. Decrease your BATCH_SIZE to be <= {len(dataset)}.")
        print("---")
        return

    # 2. Define backbone + BYOL wrapper
    resnet = resnet50(weights=None)
    resnet.fc = torch.nn.Identity()

    model = BYOL(
        backbone=resnet
    ).to(DEVICE)

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

            # --------------------------------------------------------------------
            # ✅ FIX: Call the model directly to perform the forward pass.
            # --------------------------------------------------------------------
            predictions, projections = model(x0, x1)
            p0, p1 = predictions
            z0, z1 = projections
            
            # Calculate the symmetric loss.
            # We detach the target projections (z0, z1) to stop gradients.
            loss = 0.5 * (loss_fn(p0, z1.detach()) + loss_fn(p1, z0.detach()))
            # --------------------------------------------------------------------

            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # model.update_moving_average()

        if len(dataloader) > 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    # 5. Save the final backbone
    torch.save(model.backbone.state_dict(), "byol_resnet50_backbone_patched.pth")
    print("✅ Training complete. Backbone saved to byol_resnet50_backbone_patched.pth")


if __name__ == '__main__':
    main()
