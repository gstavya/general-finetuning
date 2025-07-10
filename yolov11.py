import os
import json
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def convert_coco_to_yolo_seg(json_path, img_path, output_txt_path, category_mapping):
    """
    Convert COCO format segmentation annotations to YOLO format.
    YOLO format: class_id x1 y1 x2 y2 ... xn yn (normalized coordinates)
    """
    # Read image to get dimensions
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Read JSON annotation
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Open output file
    with open(output_txt_path, 'w') as out_f:
        # COCO format
        if 'annotations' in data:
            for ann in data['annotations']:
                if 'segmentation' in ann and ann['segmentation']:
                    # COCO category_id to YOLO class_id (0-indexed)
                    coco_cat_id = ann['category_id']
                    yolo_class_id = category_mapping.get(coco_cat_id, coco_cat_id - 1)
                    
                    # Process each polygon in segmentation
                    for polygon in ann['segmentation']:
                        if len(polygon) >= 6:  # At least 3 points
                            # Normalize coordinates
                            normalized_points = []
                            for i in range(0, len(polygon), 2):
                                x = polygon[i] / w
                                y = polygon[i+1] / h
                                # Clamp values to [0, 1]
                                x = max(0, min(1, x))
                                y = max(0, min(1, y))
                                normalized_points.extend([x, y])
                            
                            # Write to file
                            line = f"{yolo_class_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
                            out_f.write(line + '\n')
    
    return True

def load_coco_categories(json_dir):
    """
    Load category information from COCO format annotations.
    """
    categories = {}
    category_mapping = {}
    
    # Check for a main annotation file or load from first JSON
    json_files = list(Path(json_dir).glob('*.json'))
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            if 'categories' in data:
                for cat in data['categories']:
                    # Map COCO category_id to YOLO class_id (0-indexed)
                    yolo_id = len(categories)
                    categories[yolo_id] = cat['name']
                    category_mapping[cat['id']] = yolo_id
    
    # If no categories found, create default
    if not categories:
        print("Warning: No categories found in annotations. Using default.")
        categories = {0: 'object'}
        category_mapping = None
    
    return categories, category_mapping

def prepare_dataset(source_dir, output_dir):
    """
    Prepare dataset in YOLO format structure from COCO format.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # First, load categories from the dataset
    print("Loading categories from COCO annotations...")
    categories, category_mapping = load_coco_categories(source_path / 'train')
    print(f"Found categories: {categories}")
    
    # Create output directory structure
    for split in ['train', 'valid']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid']:
        split_dir = source_path / split
        
        # Get all image files
        img_files = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.jpeg')) + list(split_dir.glob('*.png'))
        
        print(f"\nProcessing {split} split: {len(img_files)} images")
        
        converted = 0
        for img_file in img_files:
            # Find corresponding JSON
            json_file = img_file.with_suffix('.json')
            
            if json_file.exists():
                # Copy image
                shutil.copy(img_file, output_path / 'images' / split / img_file.name)
                
                # Convert and save annotation
                txt_file = output_path / 'labels' / split / img_file.stem + '.txt'
                if convert_coco_to_yolo_seg(json_file, str(img_file), txt_file, category_mapping):
                    converted += 1
            else:
                print(f"Warning: No JSON found for {img_file}")
        
        print(f"Successfully converted {converted}/{len(img_files)} images in {split} split")
    
    return categories

def create_yaml_config(data_dir, class_names, yaml_path):
    """
    Create YAML configuration file for YOLOv11 training.
    """
    config = {
        'path': str(Path(data_dir).absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return yaml_path

def train_yolov11_seg():
    """
    Main function to prepare data and train YOLOv11-seg model.
    """
    # Configuration
    source_dir = "final_roboflow_2_1_1"  # Your source directory
    output_dir = "yolo_dataset"  # Processed dataset directory
    yaml_path = "dataset.yaml"  # YAML config file
    
    # Step 1: Prepare dataset
    print("=== Preparing Dataset ===")
    categories = prepare_dataset(source_dir, output_dir)
    
    # Get class names from categories
    class_names = [categories[i] for i in sorted(categories.keys())]
    
    # Step 2: Create YAML configuration
    print("\n=== Creating YAML Configuration ===")
    yaml_path = create_yaml_config(output_dir, class_names, yaml_path)
    print(f"YAML config saved to: {yaml_path}")
    
    # Step 3: Train the model
    print("\n=== Starting YOLOv11-seg Training ===")
    
    # Initialize YOLOv11-seg model (small version)
    model = YOLO('yolov11s-seg.pt')  # Download/load pretrained weights
    
    # Train the model
    results = model.train(
        data=yaml_path,           # Path to YAML config
        epochs=100,              # Number of epochs
        imgsz=640,               # Image size
        batch=16,                # Batch size (adjust based on GPU memory)
        device=0,                # GPU device (use 'cpu' if no GPU)
        workers=8,               # Number of dataloader workers
        patience=50,             # Early stopping patience
        save=True,               # Save checkpoints
        save_period=10,          # Save every N epochs
        project='runs/segment',  # Project directory
        name='yolov11s_seg',     # Experiment name
        exist_ok=True,           # Overwrite existing project/name
        pretrained=True,         # Use pretrained weights
        optimizer='auto',        # Optimizer
        verbose=True,            # Verbose output
        seed=42,                 # Random seed for reproducibility
        deterministic=True,      # Deterministic training
        single_cls=False,        # Single class training
        rect=False,              # Rectangular training
        cos_lr=False,            # Cosine LR scheduler
        close_mosaic=10,         # Disable mosaic for last N epochs
        amp=True,                # Automatic Mixed Precision training
        fraction=1.0,            # Dataset fraction to train on
        profile=False,           # Profile ONNX and TensorRT speeds
        freeze=None,             # Freeze first N layers
        
        # Augmentation parameters
        hsv_h=0.015,            # HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation augmentation
        hsv_v=0.4,              # HSV-Value augmentation
        degrees=0.0,            # Rotation augmentation
        translate=0.1,          # Translation augmentation
        scale=0.5,              # Scale augmentation
        shear=0.0,              # Shear augmentation
        perspective=0.0,        # Perspective augmentation
        flipud=0.0,             # Vertical flip augmentation
        fliplr=0.5,             # Horizontal flip augmentation
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.0,              # MixUp augmentation
        copy_paste=0.0,         # Copy-Paste augmentation (for segments)
        
        # Segmentation specific
        overlap_mask=True,       # Overlap masks during training
        mask_ratio=4,           # Mask downsample ratio
    )
    
    print("\n=== Training Complete ===")
    print(f"Best model saved to: {model.trainer.best}")
    print(f"Last model saved to: {model.trainer.last}")
    
    # Validate the model
    print("\n=== Validating Model ===")
    metrics = model.val()
    
    print("\nValidation Results:")
    print(f"Box mAP50-95: {metrics.box.map}")
    print(f"Mask mAP50-95: {metrics.seg.map}")
    
    return model

if __name__ == "__main__":
    # Run the training
    trained_model = train_yolov11_seg()
    
    # Optional: Export the model to other formats
    # trained_model.export(format='onnx')  # Export to ONNX
    # trained_model.export(format='torchscript')  # Export to TorchScript
