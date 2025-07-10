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

def download_blob(args):
    """Worker function to download a single blob."""
    blob_name, connection_string, container_name, local_dir = args
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Create local path maintaining directory structure
        local_path = os.path.join(local_dir, blob_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download blob
        with open(local_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())
        
        return f"Downloaded {blob_name}"
    except Exception as e:
        return f"Failed to download {blob_name}: {e}"

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
        
        # List all blobs
        blobs = list(container_client.list_blobs())
        if not blobs:
            print(f"Warning: No files found in container '{container_name}'.")
            return
        
        # Prepare download tasks
        tasks = [(blob.name, connection_string, container_name, local_dir) for blob in blobs]
        
        # Download in parallel
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(download_blob, tasks), total=len(tasks), desc="Downloading files"):
                pass
        
        print(f"✅ Download complete. All files saved to '{local_dir}'.")
    
    except Exception as e:
        print(f"FATAL: Error downloading from Azure: {e}")
        raise

def save_checkpoint_to_azure(trainer, connection_string, container_name, blob_name):
    """Save checkpoint directly to Azure without permanent local storage."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Ensure container exists
        try:
            container_client = blob_service_client.create_container(container_name)
            print(f"Container '{container_name}' created.")
        except Exception:
            container_client = blob_service_client.get_container_client(container_name)
        
        blob_client = container_client.get_blob_client(blob_name)
        
        # Use temporary file to save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # Save model to temporary file
            trainer.save_model(tmp_path)
            
            # Upload directly from temporary file to Azure
            print(f"Uploading checkpoint to Azure as blob: {blob_name}...")
            with open(tmp_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print("✅ Checkpoint successfully uploaded to Azure.")
            
        finally:
            # Always clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        print(f"WARNING: Failed to upload checkpoint to Azure. Error: {e}")

def download_checkpoint_from_azure(connection_string, container_name, blob_name):
    """Download checkpoint from Azure to temporary file."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Download to temporary file
        with open(tmp_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        
        return tmp_path
    except Exception as e:
        print(f"Failed to download checkpoint: {e}")
        return None

def create_data_yaml(local_data_dir):
    """Create data.yaml file for YOLO training."""
    data_yaml_path = os.path.join(local_data_dir, 'data.yaml')
    
    # Check if data.yaml already exists in the downloaded data
    if os.path.exists(data_yaml_path):
        print(f"Using existing data.yaml from {data_yaml_path}")
        # Update paths to be absolute
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths to absolute paths
        data_config['path'] = os.path.abspath(local_data_dir)
        if 'train' in data_config:
            data_config['train'] = os.path.join(os.path.abspath(local_data_dir), 'train/images')
        if 'val' in data_config:
            data_config['val'] = os.path.join(os.path.abspath(local_data_dir), 'valid/images')
        if 'test' in data_config:
            data_config['test'] = os.path.join(os.path.abspath(local_data_dir), 'test/images')
        
        # Save updated config
        updated_yaml_path = os.path.join(local_data_dir, 'data_updated.yaml')
        with open(updated_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        return updated_yaml_path
    
    else:
        # Create new data.yaml if it doesn't exist
        print("Creating new data.yaml file...")
        data_config = {
            'path': os.path.abspath(local_data_dir),
            'train': os.path.join(os.path.abspath(local_data_dir), 'train/images'),
            'val': os.path.join(os.path.abspath(local_data_dir), 'valid/images'),
            'test': os.path.join(os.path.abspath(local_data_dir), 'test/images'),
            'nc': 1,  # Number of classes
            'names': ['sidewalk']  # Class names
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        return data_yaml_path

def main():
    # --- Configuration ---
    SOURCE_DATA_CONTAINER = "testsidewalk60"
    CHECKPOINT_CONTAINER = "testyolosidewalk60"
    LOCAL_DATA_DIR = "/mnt/data/yolo_sidewalk"
    NUM_EPOCHS = 300
    BATCH_SIZE = 64  # Total batch size across all GPUs
    DEVICE = '0,1,2,3'  # Use 4 GPUs
    SAVE_PERIOD = 5
    
    # Azure connection string
    connection_string = "DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net"
    
    print(f"Using devices: {DEVICE}")
    print(f"Total batch size: {BATCH_SIZE} (will be split across 4 GPUs)")
    
    # Download data from Azure
    download_from_azure(
        connection_string=connection_string,
        container_name=SOURCE_DATA_CONTAINER,
        local_dir=LOCAL_DATA_DIR
    )
    
    # Create/update data.yaml
    data_yaml_path = create_data_yaml(LOCAL_DATA_DIR)
    print(f"Data configuration file: {data_yaml_path}")
    
    # Initialize YOLO model
    print("Initializing YOLOv11s-seg model...")
    model = YOLO('yolov11s-seg.pt')
    
    # Check for existing checkpoint in Azure
    start_epoch = 0
    latest_checkpoint = None
    temp_checkpoint_path = None
    
    print(f"Checking for existing checkpoints in Azure container '{CHECKPOINT_CONTAINER}'...")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(CHECKPOINT_CONTAINER)
        
        # List all checkpoint blobs
        checkpoints = [blob.name for blob in container_client.list_blobs() if blob.name.startswith('checkpoint_epoch_')]
        
        if checkpoints:
            # Sort to find latest
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = checkpoints[-1]
            latest_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
            
            print(f"Found checkpoint: {latest_checkpoint}")
            
            # Download checkpoint to temporary file
            temp_checkpoint_path = download_checkpoint_from_azure(
                connection_string, CHECKPOINT_CONTAINER, latest_checkpoint
            )
            
            if temp_checkpoint_path:
                # Load the checkpoint
                model = YOLO(temp_checkpoint_path)
                start_epoch = latest_epoch
                print(f"✅ Loaded checkpoint from epoch {start_epoch}")
    
    except Exception as e:
        print(f"No existing checkpoints found or error loading: {e}")
        print("Starting training from scratch...")
    
    finally:
        # Clean up temporary checkpoint file
        if temp_checkpoint_path and os.path.exists(temp_checkpoint_path):
            os.remove(temp_checkpoint_path)
    
    # Store connection string and container name for callback
    callback_config = {
        'connection_string': connection_string,
        'container_name': CHECKPOINT_CONTAINER,
        'save_period': SAVE_PERIOD
    }
    
    # Custom training callback to save checkpoints directly to Azure
    def on_epoch_end(trainer):
        """Callback to save checkpoint every SAVE_PERIOD epochs directly to Azure."""
        epoch = trainer.epoch + 1
        
        if epoch % callback_config['save_period'] == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
            
            # Save directly to Azure
            save_checkpoint_to_azure(
                trainer,
                callback_config['connection_string'],
                callback_config['container_name'],
                checkpoint_name
            )
    
    # Add callback to model
    model.add_callback("on_epoch_end", on_epoch_end)
    
    # Use temporary directory for YOLO's internal outputs
    with tempfile.TemporaryDirectory() as temp_project_dir:
        # Train the model
        print(f"\nStarting training from epoch {start_epoch + 1} to {NUM_EPOCHS}...")
        results = model.train(
            data=data_yaml_path,
            epochs=NUM_EPOCHS,
            batch=BATCH_SIZE,
            imgsz=640,
            device=DEVICE,  # Multi-GPU training
            workers=8,
            patience=50,
            save=True,
            save_period=-1,  # Disable default saving, we handle it in callback
            project=temp_project_dir,  # Use temp directory for YOLO outputs
            name='yolov11s_seg_sidewalk',
            exist_ok=True,
            pretrained=True if start_epoch == 0 else False,
            resume=True if start_epoch > 0 else False,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=True,  # Since we only have sidewalk class
            amp=True,  # Mixed precision for faster training
            
            # Augmentation parameters
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            
            # Segmentation specific
            overlap_mask=True,
            mask_ratio=4,
        )
        
        # Save final model directly to Azure
        print("\nSaving final model to Azure...")
        final_blob_name = "yolov11s_seg_sidewalk_final.pt"
        
        # Get the best model path from trainer
        best_model_path = os.path.join(temp_project_dir, 'yolov11s_seg_sidewalk', 'weights', 'best.pt')
        
        if os.path.exists(best_model_path):
            # Upload best model directly to Azure
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                blob_client = blob_service_client.get_blob_client(
                    container=CHECKPOINT_CONTAINER, 
                    blob=final_blob_name
                )
                
                with open(best_model_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"✅ Final model uploaded to Azure as: {final_blob_name}")
            except Exception as e:
                print(f"Failed to upload final model: {e}")
        
        # Validate the model
        print("\n=== Running Final Validation ===")
        metrics = model.val()
        print(f"Box mAP50-95: {metrics.box.map}")
        print(f"Mask mAP50-95: {metrics.seg.map}")
    
    print("\n✅ Training complete!")

if __name__ == '__main__':
    # Set multiprocessing start method
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    main()
