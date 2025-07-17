import os
import io
import yaml
import torch
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from ultralytics import YOLO
import multiprocessing
from tqdm import tqdm
import tempfile
import warnings
import shutil
import json
import time
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import settings
settings.update({'wandb': True})


# Set YOLO config directory to avoid permission warnings
os.environ['YOLO_CONFIG_DIR'] = '/tmp/yolo_config'
os.environ['WANDB_API_KEY'] = 'dab85e8256791ddd93e4b37ecd163130376f7ffc'

wandb.login(key="dab85e8256791ddd93e4b37ecd163130376f7ffc")

wandb.init(project="ultralytics", name="yolo-sidewalk-training")

# Suppress Ultralytics warnings during multiprocessing
warnings.filterwarnings('ignore', message='user config directory')
warnings.filterwarnings('ignore', message='Error decoding JSON')

# --- Helper functions (adjusted for minimal output and no external saving/Wandb) ---

import albumentations as A
import cv2
from pathlib import Path
import numpy as np
import shutil

def augment_dataset_10x_yolo_format(source_dir, output_dir, num_augmentations=9):
    """Create 9 augmented versions of training images only, maintaining YOLO structure"""
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        Path(output_dir, split, 'images').mkdir(parents=True, exist_ok=True)
        Path(output_dir, split, 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy data.yaml if it exists
    source_yaml = Path(source_dir) / 'data.yaml'
    if source_yaml.exists():
        shutil.copy(source_yaml, Path(output_dir) / 'data.yaml')
    
    # Process each split
    for split in ['train', 'val', 'test']:
        images_dir = Path(source_dir) / split / 'images'
        labels_dir = Path(source_dir) / split / 'labels'
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue
            
        print(f"Processing {split} split...")
        
        # Get all images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for img_path in image_files:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read {img_path}, skipping...")
                continue
                
            # Get corresponding label file
            label_path = labels_dir / img_path.with_suffix('.txt').name
            
            # Save original to new location
            new_img_path = Path(output_dir) / split / 'images' / img_path.name
            cv2.imwrite(str(new_img_path), image)
            
            # Save original labels
            if label_path.exists():
                new_label_path = Path(output_dir) / split / 'labels' / label_path.name
                shutil.copy(label_path, new_label_path)
            
            # Only augment training images
            if split == 'train':
                # Define transform without bbox parameters
                transform = A.Compose([
                    A.RandomRotate90(p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Transpose(p=0.3),
                    A.OneOf([
                        A.GaussNoise(p=1),
                        A.GaussianBlur(p=1),
                        A.MotionBlur(p=1),
                    ], p=0.3),
                    A.OneOf([
                        A.OpticalDistortion(p=1),
                        A.GridDistortion(p=1),
                        A.ElasticTransform(p=1),
                    ], p=0.3),
                    A.OneOf([
                        A.CLAHE(p=1),
                        A.RandomBrightnessContrast(p=1),
                        A.RandomGamma(p=1),
                    ], p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
                    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                    A.RandomShadow(p=0.3),
                    A.RandomFog(p=0.3),
                ])
                
                # Generate augmented versions
                for i in range(num_augmentations):
                    try:
                        # Apply augmentation to image only
                        augmented = transform(image=image)
                        
                        # Save augmented image
                        aug_img_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                        aug_img_path = Path(output_dir) / 'train' / 'images' / aug_img_name
                        cv2.imwrite(str(aug_img_path), augmented['image'])
                        
                        # Copy labels as-is (keeping invalid boxes)
                        if label_path.exists():
                            aug_label_path = Path(output_dir) / 'train' / 'labels' / f"{img_path.stem}_aug{i}.txt"
                            shutil.copy(label_path, aug_label_path)
                            
                    except Exception as e:
                        print(f"Warning: Augmentation failed for {img_path}: {e}")
                        continue
        
        if split == 'train':
            print(f"Created {len(image_files) * (num_augmentations + 1)} training images from {len(image_files)} originals")
        else:
            print(f"Copied {len(image_files)} {split} images")

def download_blob(args):
    """Worker function to download a single blob."""
    blob_name, connection_string, container_name, local_dir = args

    # Suppress warnings in worker processes
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['YOLO_CONFIG_DIR'] = '/tmp/yolo_config'

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        local_path = os.path.join(local_dir, blob_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())

        return True # Indicate success
    except Exception as e:
        print(f"Failed to download {blob_name}: {e}") # Keep minimal error output
        return False # Indicate failure

def download_from_azure(connection_string, container_name, local_dir):
    """Download all files from Azure container to local directory."""
    if os.path.exists(local_dir) and len(list(Path(local_dir).rglob('*'))) > 100:
        print(f"Data already exists in '{local_dir}'. Skipping download.")
        return

    print(f"Downloading data from Azure container '{container_name}' to '{local_dir}'...")
    os.makedirs(local_dir, exist_ok=True)

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        blobs = list(container_client.list_blobs())
        if not blobs:
            print(f"Warning: No files found in container '{container_name}'.")
            return

        tasks = [(blob.name, connection_string, container_name, local_dir) for blob in blobs]

        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            # Use tqdm to show progress, but results are just True/False
            for _ in tqdm(pool.imap_unordered(download_blob, tasks), total=len(tasks), desc="Downloading files"):
                pass

        print(f"✅ Download complete. All files saved to '{local_dir}'.")

    except Exception as e:
        print(f"FATAL: Error downloading from Azure: {e}") # Keep fatal error output
        raise

def download_and_extract_checkpoint(connection_string, container_name, blob_name, extract_dir):
    """
    Download and extract checkpoint from Azure.
    This function remains as it loads a model for resumption.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        tar_path = os.path.join(extract_dir, blob_name)
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)

        print(f"Downloading checkpoint '{blob_name}'...") # Keep minimal download print
        with open(tar_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        import tarfile
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_dir)

        os.remove(tar_path)

        # Ultralytics checkpoints are typically saved as directories containing weights and other files.
        extracted_content_path = os.path.join(extract_dir, Path(blob_name).stem)
        if os.path.isdir(extracted_content_path):
            potential_weights_path = os.path.join(extracted_content_path, 'weights', 'last.pt')
            if os.path.exists(potential_weights_path):
                return potential_weights_path
            elif os.path.exists(os.path.join(extracted_content_path, 'last.pt')): # Sometimes directly in folder
                return os.path.join(extracted_content_path, 'last.pt')
        elif os.path.exists(os.path.join(extract_dir, Path(blob_name).stem + '.pt')): # For direct .pt tarred
             return os.path.join(extract_dir, Path(blob_name).stem + '.pt')

        print(f"WARNING: Could not find 'last.pt' or similar in extracted checkpoint from {blob_name}.")
        return None

    except Exception as e:
        print(f"Failed to download/extract checkpoint: {e}") # Keep minimal error output
        return None

def create_data_yaml(local_data_dir):
    """Create data.yaml file for YOLO training."""
    data_yaml_path = os.path.join(local_data_dir, 'data.yaml')

    if os.path.exists(data_yaml_path):
        print(f"Using existing data.yaml from {data_yaml_path}")
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        data_config['path'] = os.path.abspath(local_data_dir)
        if 'train' in data_config:
            data_config['train'] = os.path.join(os.path.abspath(local_data_dir), 'train/images')
        if 'val' in data_config:
            data_config['val'] = os.path.join(os.path.abspath(local_data_dir), 'val/images')
        if 'test' in data_config:
            data_config['test'] = os.path.join(os.path.abspath(local_data_dir), 'test/images')

        updated_yaml_path = os.path.join(local_data_dir, 'data_updated.yaml')
        with open(updated_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        return updated_yaml_path

    else:
        print("Creating new data.yaml file...")
        data_config = {
            'path': os.path.abspath(local_data_dir),
            'train': os.path.join(os.path.abspath(local_data_dir), 'train/images'),
            'val': os.path.join(os.path.abspath(local_data_dir), 'val/images'),
            'test': os.path.join(os.path.abspath(local_data_dir), 'test/images'),
            'nc': 1,
            'names': ['sidewalk']
        }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        return data_yaml_path

# Removed save_full_checkpoint_to_azure function entirely

def main():
    # --- Configuration ---
    SOURCE_DATA_CONTAINER = "13ksidewalk60"
    CHECKPOINT_CONTAINER = "13ksidewalk60yolo"
    LOCAL_DATA_DIR = "/mnt/data/yolo_sidewalk"
    NUM_EPOCHS = 10
    BATCH_SIZE = 512
    DEVICE = '0,1,2,3'

    # Create YOLO config directory
    os.makedirs('/tmp/yolo_config', exist_ok=True)

    # Azure connection string
    connection_string = "DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net"

    print(f"Using devices: {DEVICE}")
    print(f"Total batch size: {BATCH_SIZE} (will be split across 4 GPUs)")

    # Removed Wandb initialization

    # Download data from Azure
    download_from_azure(
        connection_string=connection_string,
        container_name=SOURCE_DATA_CONTAINER,
        local_dir=LOCAL_DATA_DIR
    )

    # Create/update data.yaml
    data_yaml_path = create_data_yaml(LOCAL_DATA_DIR)
    print(f"Data configuration file: {data_yaml_path}")

    # augment_dataset_10x_yolo_format(
    #     source_dir="/mnt/data/yolo_sidewalk",
    #     output_dir="/mnt/data/yolo_sidewalk_10x"
    # )

    # data_yaml_path = "/mnt/data/yolo_sidewalk_10x/data.yaml"
    # print(f"Data configuration file: {data_yaml_path}")

    model = YOLO('yolo11n-seg.pt')
    add_wandb_callback(model, enable_model_checkpointing=True)

    results = model.train(
        data=data_yaml_path, 
        epochs=NUM_EPOCHS, 
        imgsz=640, 
        plots=True, 
        val=True,
        project="ultralytics",
        name="yolo-sidewalk-run",
        hsv_h=0.015,       # Hue (subtle color shift)
        hsv_s=0.7,         # Saturation (strong color intensity)
        hsv_v=0.4,         # Value (moderate brightness)
        
        degrees=180.0,      # Rotation (+/- 30 degrees)
        translate=0.75,     # Translation (+/- 20%)
        scale=1.5,         # Scale/Zoom (+/- 70%)
        shear=15,        # Shear (+/- 15 degrees)
        perspective=0.0005, # Perspective distortion
        
        flipud=0.5,        # Flip image up-down
        fliplr=0.5,        # Flip image left-right
        
        mosaic=1.0,        # Mosaic composition
        mixup=0.5,         # Mixup composition
        copy_paste=0.5     # Copy-paste for segmentation
    )
    print("\nSaving final model to Azure...")

    best_model_path = Path(results.save_dir, "weights", "best.pt")
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    for i in range(NUM_EPOCHS):
        if i % 5 == 0:
            model_path = Path(results.save_dir, "weights", f"epoch{i}.pt")
            if model_path.exists():
                blob_client = blob_service_client.get_blob_client(
                    container=CHECKPOINT_CONTAINER,
                    blob=f"yolo11n_seg_sidewalk_epoch{i}.pt"
                )
                with open(model_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"✅ Epoch {i} model uploaded to Azure")
    
    # Upload best model
    if best_model_path.exists():
        blob_client = blob_service_client.get_blob_client(
            container=CHECKPOINT_CONTAINER,
            blob="yolo11n_seg_sidewalk_final_best.pt"
        )
        with open(best_model_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"✅ Best model uploaded to Azure")

    print("\n✅ Training complete! Final model saved.")
    wandb.finish()

if __name__ == '__main__':
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    main()
