import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
import cv2
from collections import defaultdict
from azure.storage.blob import BlobServiceClient
from torch.cuda.amp import GradScaler, autocast

# ---------- 0. AZURE BLOB STORAGE UTILITY ---------------------------
def download_data_from_azure(connection_string, container_name, blob_prefix, local_download_path):
    """
    Downloads data from a specified folder in Azure Blob Storage to a local directory.

    Args:
        connection_string (str): The connection string for the Azure Storage account.
        container_name (str): The name of the blob container.
        blob_prefix (str): The prefix (folder path) to filter blobs.
        local_download_path (str): The local directory to save files to.
    """
    try:
        print("Connecting to Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        print(f"Successfully connected to container '{container_name}'.")

        print(f"Listing blobs with prefix '{blob_prefix}'...")
        blob_list = container_client.list_blobs(name_starts_with=blob_prefix)
        
        for blob in blob_list:
            # Construct the full local path
            relative_path = os.path.relpath(blob.name, blob_prefix)
            local_file_path = os.path.join(local_download_path, 'roads', relative_path)
            local_file_dir = os.path.dirname(local_file_path)

            # Create local directory if it doesn't exist
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            print(f"Downloading '{blob.name}' to '{local_file_path}'...")
            blob_client = container_client.get_blob_client(blob.name)
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        
        print("--- Azure data download complete. ---")

    except Exception as e:
        print(f"An error occurred during Azure download: {e}")
        raise

# ---------- 1. POLYGON DATASET ---------------------------
class CocoPolygonDataset(Dataset):
    """
    COCO Polygon Detection Dataset.
    Returns images and polygon annotations for object detection.
    """
    def __init__(self, image_dir, ann_file, transforms=None, num_vertices=8, input_size=512):
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.num_vertices = num_vertices
        self.input_size = input_size
        self.stride_levels = [4, 8, 16, 32]

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        self.coco = COCO(ann_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_map = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

    def __len__(self):
        return len(self.image_ids)

    def _sample_polygon_vertices(self, segmentation, num_vertices):
        """Sample fixed number of vertices from polygon segmentation."""
        if not segmentation:
            return np.zeros((num_vertices, 2))
        
        poly = np.array(segmentation[0]).reshape(-1, 2)
        
        if len(poly) < num_vertices:
            indices = np.linspace(0, len(poly) - 1, num_vertices)
            sampled_poly = np.array([poly[int(idx)] for idx in indices])
        else:
            step = len(poly) // num_vertices
            sampled_poly = poly[::step][:num_vertices]
        
        return sampled_poly

    def _create_target_tensors(self, img_w, img_h, annotations):
        """Create target tensors for each feature map level."""
        targets = []
        
        for stride in self.stride_levels:
            feat_h, feat_w = self.input_size // stride, self.input_size // stride
            
            obj_target = torch.zeros((feat_h, feat_w, 1))
            cls_target = torch.zeros((feat_h, feat_w, self.num_classes))
            bbox_target = torch.zeros((feat_h, feat_w, 4))
            poly_target = torch.zeros((feat_h, feat_w, 2 * self.num_vertices))
            
            for ann in annotations:
                bbox = ann['bbox']
                cx, cy = (bbox[0] + bbox[2] / 2) / img_w, (bbox[1] + bbox[3] / 2) / img_h
                grid_x, grid_y = int(cx * feat_w), int(cy * feat_h)
                
                if 0 <= grid_x < feat_w and 0 <= grid_y < feat_h:
                    obj_target[grid_y, grid_x, 0] = 1.0
                    cls_id = self.cat_id_map[ann['category_id']]
                    cls_target[grid_y, grid_x, cls_id] = 1.0
                    bbox_target[grid_y, grid_x] = torch.tensor([cx, cy, bbox[2] / img_w, bbox[3] / img_h])
                    
                    if 'segmentation' in ann:
                        poly = self._sample_polygon_vertices(ann['segmentation'], self.num_vertices)
                        poly_norm = poly / np.array([img_w, img_h])
                        poly_target[grid_y, grid_x] = torch.tensor(poly_norm.flatten())
            
            targets.append({'obj': obj_target, 'cls': cls_target, 'bbox': bbox_target, 'poly': poly_target})
        
        return targets

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found {img_path}, skipping.")
            return None

        try:
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            targets = self._create_target_tensors(img_w, img_h, anns)
            if self.transforms:
                img = self.transforms(img)
            return img, targets
        except Exception as e:
            print(f"Error loading image or annotation {img_path}: {e}. Skipping.")
            return None

# ---------- 2. POLYYOLO MODEL (UNCHANGED) ---------------------------
class PolyYOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_vertices=8, fpn_dim=256):
        super().__init__()
        self.fpn_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, fpn_dim, 1, bias=False), nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True)) for in_ch in in_channels])
        self.num_classes, self.num_vertices = num_classes, num_vertices
        self.cls_head, self.bbox_head, self.poly_head, self.obj_head = (self._make_head(fpn_dim, out_ch) for out_ch in [num_classes, 4, 2 * num_vertices, 1])
        self.strides = [4, 8, 16, 32]
    def _make_head(self, in_channels, out_channels): return nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, 1))
    def forward(self, features):
        outputs = []
        for i, (feat, fpn_conv) in enumerate(zip(features, self.fpn_convs)):
            fpn_feat = fpn_conv(feat)
            cls_pred, bbox_pred, poly_pred, obj_pred = (head(fpn_feat) for head in [self.cls_head, self.bbox_head, self.poly_head, self.obj_head])
            batch_size, _, h, w = cls_pred.shape
            outputs.append({ 'cls': cls_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, h * w, self.num_classes), 'bbox': bbox_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, h * w, 4), 'poly': poly_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, h * w, 2 * self.num_vertices), 'obj': obj_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, h * w, 1), 'feat_h': h, 'feat_w': w, 'stride': self.strides[i] })
        return outputs

class PolyYOLOModel(nn.Module):
    def __init__(self, backbone, num_classes, num_vertices=8):
        super().__init__()
        self.backbone = create_feature_extractor(backbone, return_nodes={'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'})
        self.head = PolyYOLOHead(in_channels=[256, 512, 1024, 2048], num_classes=num_classes, num_vertices=num_vertices)
    def forward(self, x): return self.head(list(self.backbone(x).values()))

# ---------- 3. LOSS FUNCTION (UNCHANGED) ---------------------------
class PolyYOLOLoss(nn.Module):
    def __init__(self, num_classes, num_vertices):
        super().__init__()
        self.num_classes, self.num_vertices = num_classes, num_vertices
        self.bce_obj, self.bce_cls, self.smooth_l1 = nn.BCEWithLogitsLoss(reduction='none'), nn.BCEWithLogitsLoss(reduction='none'), nn.SmoothL1Loss(reduction='none')
    def forward(self, predictions, targets):
        device, total_loss, losses = predictions[0]['cls'].device, torch.tensor(0.0, device=device), defaultdict(float)
        for pred, target_batch in zip(predictions, targets):
            batch_size = pred['cls'].shape[0]
            obj_target, cls_target, bbox_target, poly_target = (target_batch[k].to(device).view(batch_size, -1, v) for k, v in {'obj': 1, 'cls': self.num_classes, 'bbox': 4, 'poly': 2 * self.num_vertices}.items())
            obj_loss, obj_mask = self.bce_obj(pred['obj'], obj_target), obj_target.squeeze(-1) > 0.5
            if obj_mask.sum() > 0:
                losses['cls'] += self.bce_cls(pred['cls'][obj_mask], cls_target[obj_mask]).mean()
                losses['bbox'] += self.smooth_l1(pred['bbox'][obj_mask], bbox_target[obj_mask]).mean()
                losses['poly'] += self.smooth_l1(pred['poly'][obj_mask], poly_target[obj_mask]).mean()
            losses['obj'] += obj_loss.mean()
        total_loss = losses['obj'] + losses['cls'] + losses['bbox'] * 5.0 + losses['poly'] * 10.0
        return total_loss, losses

# ---------- 4. COLLATE FUNCTION ---------------------------
def collate_fn(batch):
    """Filters out None items and collates the batch."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None
    images, all_targets = zip(*batch)
    images = torch.stack(images, 0)
    batched_targets = []
    for i in range(len(all_targets[0])):
        batched_target_level = {key: torch.stack([t[i][key] for t in all_targets], 0) for key in all_targets[0][i]}
        batched_targets.append(batched_target_level)
    return images, batched_targets

# ---------- 5. MAIN TRAINING FUNCTION ---------------------------
def main():
    """Main function to run the training pipeline with GPU optimizations."""
    # --- AZURE & LOCAL PATH CONFIGURATION ---
    AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net"
    AZURE_CONTAINER_NAME = 'labeled'
    AZURE_BLOB_PREFIX = 'sidewalk/'
    LOCAL_DATA_PATH = '/mnt/data'
    MODEL_SAVE_PATH = '/mnt/satellite-resnet'
    
    # Download data from Azure
    download_data_from_azure(AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, AZURE_BLOB_PREFIX, LOCAL_DATA_PATH)

    # --- GPU OPTIMIZED CONFIGURATION ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    print(f"Using device: {device}. Automatic Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")

    NUM_VERTICES = 8
    BATCH_SIZE = 256  # Increased for GPU
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4  # Increased for faster data loading
    
    image_transforms = T.Compose([
        T.Resize((512, 512)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    BASE_DIR = os.path.join(LOCAL_DATA_PATH, 'sidewalk')
    train_dataset = CocoPolygonDataset(os.path.join(BASE_DIR, 'train'), os.path.join(BASE_DIR, 'train_annotations.coco.json'), image_transforms, num_vertices=NUM_VERTICES)
    val_dataset = CocoPolygonDataset(os.path.join(BASE_DIR, 'valid'), os.path.join(BASE_DIR, 'valid_annotations.coco.json'), image_transforms, num_vertices=NUM_VERTICES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    print("Initializing PolyYOLO model...")
    num_classes = train_dataset.num_classes
    backbone = torchvision.models.resnet50(weights='IMAGENET1K_V1') # Start with pre-trained weights
    model = PolyYOLOModel(backbone=backbone, num_classes=num_classes, num_vertices=NUM_VERTICES).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = PolyYOLOLoss(num_classes=num_classes, num_vertices=NUM_VERTICES)
    scaler = GradScaler(enabled=use_amp)

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            if images is None: continue
            images = images.to(device, non_blocking=True)
            
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            if (i + 1) % 10 == 0: print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                if images is None: continue
                images = images.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    outputs = model(images)
                    loss, _ = criterion(outputs, targets)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"epoch_{epoch+1}.pth")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': avg_val_loss}, checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")
    
    print("\n--- Training completed! ---")

if __name__ == '__main__':
    main()
