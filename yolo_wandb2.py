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

# Set YOLO config directory to avoid permission warnings
os.environ['YOLO_CONFIG_DIR'] = '/tmp/yolo_config'

# Suppress Ultralytics warnings during multiprocessing
warnings.filterwarnings('ignore', message='user config directory')
warnings.filterwarnings('ignore', message='Error decoding JSON')

# --- Helper functions (adjusted for minimal output and no external saving/Wandb) ---

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
    CHECKPOINT_CONTAINER = "13ksidewalk60yolotest"
    LOCAL_DATA_DIR = "/mnt/data/yolo_sidewalk"
    NUM_EPOCHS = 50
    BATCH_SIZE = 256
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

    # Initialize variables for checkpoint loading
    start_epoch = 0
    resume_path = False
    model = None

    # Check for existing checkpoint in Azure (for resumption)
    print(f"Checking for existing checkpoints in Azure container '{CHECKPOINT_CONTAINER}' for resumption...")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(CHECKPOINT_CONTAINER)

        latest_checkpoint_to_load = None
        
        # Prioritize 'latest_resume_checkpoint.pth' if present for direct loading
        latest_pt_blob = next((blob.name for blob in container_client.list_blobs() if blob.name == 'latest_resume_checkpoint.pth'), None)
        
        if latest_pt_blob:
            latest_checkpoint_to_load = latest_pt_blob
            print(f"Found 'latest_resume_checkpoint.pth' for direct resumption.")
        else:
            # Fallback to finding the highest epoch tar file if no direct .pt blob
            checkpoint_tar_blobs = [blob.name for blob in container_client.list_blobs() if blob.name.startswith('checkpoint_epoch_') and blob.name.endswith('.tar')]
            if checkpoint_tar_blobs:
                checkpoint_tar_blobs.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                latest_checkpoint_to_load = checkpoint_tar_blobs[-1]
                print(f"Found latest epoch checkpoint tar: {latest_checkpoint_to_load}")
        
        if latest_checkpoint_to_load:
            with tempfile.TemporaryDirectory() as temp_dir_for_resume:
                extracted_weights_path = download_and_extract_checkpoint(
                    connection_string, CHECKPOINT_CONTAINER, latest_checkpoint_to_load, temp_dir_for_resume
                )
                if extracted_weights_path and os.path.exists(extracted_weights_path):
                    model = YOLO(extracted_weights_path)
                    # Try to infer start_epoch if from a tar file, assuming naming convention
                    if latest_checkpoint_to_load.startswith('checkpoint_epoch_'):
                        try:
                            start_epoch = int(latest_checkpoint_to_load.split('_')[2].split('.')[0])
                            print(f"Resuming training from epoch {start_epoch}.")
                        except ValueError:
                            print("Could not parse epoch from checkpoint name. Resuming from 0.")
                    resume_path = extracted_weights_path # Set resume path for Ultralytics
                else:
                    print("Failed to get weights path from extracted checkpoint. Starting from scratch...")
                    model = YOLO('yolov11n-seg.pt')
        else:
            print("No checkpoints found, starting from scratch...")
            model = YOLO('yolov11n-seg.pt')

    except Exception as e:
        print(f"Error checking for checkpoints: {e}")
        print("Starting training from scratch...")
        model = YOLO('yolov11n-seg.pt')

    add_wandb_callback(model, enable_model_checkpointing=True)

    with tempfile.TemporaryDirectory() as temp_project_dir:
        remaining_epochs = NUM_EPOCHS - start_epoch

        if remaining_epochs <= 0:
            print(f"Training already completed ({start_epoch} epochs done, target was {NUM_EPOCHS})")
            return

        print(f"\nStarting training from epoch {start_epoch + 1} to {NUM_EPOCHS}...")
        print(f"Training for {remaining_epochs} more epochs...")

        train_args = {
            'data': data_yaml_path,
            'epochs': remaining_epochs,
            'batch': BATCH_SIZE,
            'imgsz': 640,
            'device': DEVICE,
            'workers': 8,
            'save': True, # Keep True so Ultralytics saves 'last.pt' and 'best.pt'
            'save_period': -1, # Disable periodic saving by Ultralytics
            'project': temp_project_dir, # Use a temporary directory for output
            'name': 'yolov11n_seg_sidewalk_run', # A simple name for Ultralytics' internal run folder
            'exist_ok': True,
            'pretrained': False,
            'resume': resume_path if resume_path else False, # Pass path for resuming
            'optimizer': 'auto',
            'verbose': False, # Set verbose to False to suppress training loss output
            'seed': 42,
            'deterministic': True,
            'single_cls': True,
            'amp': True,
            'plots': False, # Disable plotting to reduce output and dependencies
            # Augmentation parameters
            'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1,
            'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
            'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0,
            # Segmentation specific
            'overlap_mask': True, 'mask_ratio': 4,
        }

        # Train the model silently, only the final result is of interest
        results = model.train(**train_args)

        # --- Save only the final model to Azure ---
        print("\nSaving final model to Azure...")

        final_blob_name = "yolov11n_seg_sidewalk_final.pt"
        # Ultralytics saves the last model in its run directory, specifically:
        # {temp_project_dir}/{name}/weights/last.pt
        final_model_path_local = os.path.join(temp_project_dir, train_args['name'], 'weights', 'last.pt')
        
        if os.path.exists(final_model_path_local):
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                blob_client = blob_service_client.get_blob_client(
                    container=CHECKPOINT_CONTAINER,
                    blob=final_blob_name
                )
                with open(final_model_path_local, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"✅ Final trained model uploaded to Azure as: {final_blob_name}")
            except Exception as e:
                print(f"Failed to upload final model to Azure: {e}")
        else:
            print(f"❌ Error: Final model weights not found at {final_model_path_local}.")

    print("\n✅ Training complete! Final model saved.")
    wandb.finish()

if __name__ == '__main__':
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    main()
