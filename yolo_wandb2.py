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
    NUM_EPOCHS = 300
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

    model = YOLO('yolo11x-seg.pt')
    add_wandb_callback(model, enable_model_checkpointing=True)

    results = model.train(
        data=data_yaml_path, 
        epochs=NUM_EPOCHS, 
        imgsz=640, 
        plots=True, 
        val=True,
        project="ultralytics",
        name="yolo-sidewalk-run"
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
                    blob=f"yolo11x_seg_sidewalk_epoch{i}.pt"
                )
                with open(model_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"✅ Epoch {i} model uploaded to Azure")
    
    # Upload best model
    if best_model_path.exists():
        blob_client = blob_service_client.get_blob_client(
            container=CHECKPOINT_CONTAINER,
            blob="yolo11x_seg_sidewalk_final_best.pt"
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
