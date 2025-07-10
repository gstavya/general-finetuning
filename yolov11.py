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

# Set YOLO config directory to avoid permission warnings
os.environ['YOLO_CONFIG_DIR'] = '/tmp/yolo_config'

# Suppress Ultralytics warnings during multiprocessing
warnings.filterwarnings('ignore', message='user config directory')
warnings.filterwarnings('ignore', message='Error decoding JSON')

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

def save_full_checkpoint_to_azure(trainer, connection_string, container_name, epoch):
    """Save complete checkpoint with all training state to Azure."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Ensure container exists
        try:
            container_client = blob_service_client.create_container(container_name)
            print(f"Container '{container_name}' created.")
        except Exception:
            container_client = blob_service_client.get_container_client(container_name)
        
        # Create temporary directory for checkpoint files
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, f'checkpoint_epoch_{epoch}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save the model weights
            weights_path = os.path.join(checkpoint_dir, 'weights.pt')
            trainer.save_model(weights_path)
            
            # Save optimizer state
            optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
            torch.save(trainer.optimizer.state_dict(), optimizer_path)
            
            # Save training args and state
            training_state = {
                'epoch': epoch,
                'best_fitness': trainer.best_fitness,
                'fitness': trainer.fitness,
                'ema': trainer.ema.state_dict() if hasattr(trainer, 'ema') and trainer.ema else None,
                'updates': trainer.optimizer.state_dict()['state'].get(0, {}).get('step', 0) if trainer.optimizer else 0,
                'train_args': vars(trainer.args),
            }
            
            state_path = os.path.join(checkpoint_dir, 'training_state.pt')
            torch.save(training_state, state_path)
            
            # Save results CSV if exists
            if hasattr(trainer, 'csv') and trainer.csv and os.path.exists(trainer.csv):
                shutil.copy(trainer.csv, os.path.join(checkpoint_dir, 'results.csv'))
            
            # Create a tar archive of the checkpoint directory
            tar_path = os.path.join(temp_dir, f'checkpoint_epoch_{epoch}.tar')
            shutil.make_archive(tar_path.replace('.tar', ''), 'tar', checkpoint_dir)
            
            # Upload the tar file to Azure
            blob_name = f'checkpoint_epoch_{epoch}.tar'
            blob_client = container_client.get_blob_client(blob_name)
            
            print(f"Uploading complete checkpoint to Azure as: {blob_name}...")
            with open(tar_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"✅ Checkpoint for epoch {epoch} successfully uploaded to Azure.")
    
    except Exception as e:
        print(f"WARNING: Failed to upload checkpoint to Azure. Error: {e}")

def download_and_extract_checkpoint(connection_string, container_name, blob_name, extract_dir):
    """Download and extract checkpoint from Azure."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Download tar file
        tar_path = os.path.join(extract_dir, blob_name)
        with open(tar_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        
        # Extract tar file
        import tarfile
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_dir)
        
        # Remove tar file
        os.remove(tar_path)
        
        # Return path to extracted checkpoint directory
        checkpoint_name = blob_name.replace('.tar', '')
        return os.path.join(extract_dir, checkpoint_name)
    
    except Exception as e:
        print(f"Failed to download/extract checkpoint: {e}")
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
    
    # Create YOLO config directory
    os.makedirs('/tmp/yolo_config', exist_ok=True)
    
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
    
    # Initialize variables
    start_epoch = 0
    resume_path = None
    training_state = None
    
    # Check for existing checkpoint in Azure
    print(f"Checking for existing checkpoints in Azure container '{CHECKPOINT_CONTAINER}'...")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(CHECKPOINT_CONTAINER)
        
        # List all checkpoint blobs
        checkpoints = [blob.name for blob in container_client.list_blobs() if blob.name.startswith('checkpoint_epoch_') and blob.name.endswith('.tar')]
        
        if checkpoints:
            # Sort to find latest
            checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            latest_checkpoint = checkpoints[-1]
            latest_epoch = int(latest_checkpoint.split('_')[2].split('.')[0])
            
            print(f"Found checkpoint: {latest_checkpoint}")
            
            # Download and extract checkpoint
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_dir = download_and_extract_checkpoint(
                    connection_string, CHECKPOINT_CONTAINER, latest_checkpoint, temp_dir
                )
                
                if checkpoint_dir:
                    # Load weights
                    weights_path = os.path.join(checkpoint_dir, 'weights.pt')
                    if os.path.exists(weights_path):
                        model = YOLO(weights_path)
                        print(f"✅ Loaded model weights from epoch {latest_epoch}")
                        
                        # Load training state
                        state_path = os.path.join(checkpoint_dir, 'training_state.pt')
                        if os.path.exists(state_path):
                            training_state = torch.load(state_path)
                            start_epoch = training_state['epoch']
                            print(f"✅ Loaded training state from epoch {start_epoch}")
                        
                        # Copy weights to a persistent location for resume
                        resume_path = '/tmp/resume_weights.pt'
                        shutil.copy(weights_path, resume_path)
                    else:
                        print("Weights file not found in checkpoint, starting from scratch...")
                        model = YOLO('yolo11s-seg.pt')
                else:
                    print("Failed to extract checkpoint, starting from scratch...")
                    model = YOLO('yolo11s-seg.pt')
        else:
            print("No checkpoints found, starting from scratch...")
            model = YOLO('yolo11s-seg.pt')
    
    except Exception as e:
        print(f"Error checking for checkpoints: {e}")
        print("Starting training from scratch...")
        model = YOLO('yolo11s-seg.pt')
    
    # Store connection string and container name for callback
    callback_config = {
        'connection_string': connection_string,
        'container_name': CHECKPOINT_CONTAINER,
        'save_period': SAVE_PERIOD,
        'data_yaml': data_yaml_path,
        'start_epoch': start_epoch
    }
    
    # Custom training callback to save complete checkpoints
    def on_train_epoch_end(trainer):
        """Callback to save complete checkpoint and display metrics."""
        # Calculate actual epoch considering resume
        actual_epoch = callback_config['start_epoch'] + trainer.epoch + 1
        
        # Display validation metrics
        print(f"\n--- Epoch {actual_epoch} Validation Metrics ---")
        try:
            metrics = trainer.model.val(data=callback_config['data_yaml'], verbose=False)
            
            print(f"Box Metrics:")
            print(f"  Precision: {metrics.box.p.mean():.4f}")
            print(f"  Recall: {metrics.box.r.mean():.4f}")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            
            print(f"Mask Metrics:")
            print(f"  Precision: {metrics.seg.p.mean():.4f}")
            print(f"  Recall: {metrics.seg.r.mean():.4f}")
            print(f"  mAP50: {metrics.seg.map50:.4f}")
            print(f"  mAP50-95: {metrics.seg.map:.4f}")
            print("------------------------------------\n")
        except Exception as e:
            print(f"Error calculating validation metrics: {e}")
            print("------------------------------------\n")
        
        # Save checkpoint every SAVE_PERIOD epochs
        if actual_epoch % callback_config['save_period'] == 0:
            save_full_checkpoint_to_azure(
                trainer,
                callback_config['connection_string'],
                callback_config['container_name'],
                actual_epoch
            )
    
    # Add callback to model
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # Use temporary directory for YOLO's internal outputs
    with tempfile.TemporaryDirectory() as temp_project_dir:
        # Calculate remaining epochs
        remaining_epochs = NUM_EPOCHS - start_epoch
        
        if remaining_epochs <= 0:
            print(f"Training already completed ({start_epoch} epochs done, target was {NUM_EPOCHS})")
            return
        
        # Train the model
        print(f"\nStarting training from epoch {start_epoch + 1} to {NUM_EPOCHS}...")
        print(f"Training for {remaining_epochs} more epochs...")
        
        # Prepare training arguments
        train_args = {
            'data': data_yaml_path,
            'epochs': remaining_epochs,  # Train only remaining epochs
            'batch': BATCH_SIZE,
            'imgsz': 640,
            'device': DEVICE,
            'workers': 8,
            'patience': 0,  # Disable early stopping
            'save': True,
            'save_period': -1,  # Disable default saving
            'project': temp_project_dir,
            'name': 'yolov11s_seg_sidewalk',
            'exist_ok': True,
            'pretrained': False,  # Already have a model
            'resume': resume_path if resume_path and os.path.exists(resume_path) else False,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': True,
            'amp': True,
            
            # Augmentation parameters
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            
            # Segmentation specific
            'overlap_mask': True,
            'mask_ratio': 4,
        }
        
        # If we have training state and optimizer state, we need to properly resume
        if training_state and resume_path:
            # YOLO's resume functionality should handle optimizer state loading
            train_args['resume'] = resume_path
        
        results = model.train(**train_args)
        
        # Save final model and checkpoint
        print("\nSaving final model to Azure...")
        
        # Save as both final model and final checkpoint
        final_blob_name = "yolov11s_seg_sidewalk_final.pt"
        best_model_path = os.path.join(temp_project_dir, 'yolov11s_seg_sidewalk', 'weights', 'best.pt')
        
        if os.path.exists(best_model_path):
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                blob_client = blob_service_client.get_blob_client(
                    container=CHECKPOINT_CONTAINER, 
                    blob=final_blob_name
                )
                
                with open(best_model_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"✅ Final model uploaded to Azure as: {final_blob_name}")
                
                # Also save a complete final checkpoint
                # Get trainer from the model's last training session
                if hasattr(model, 'trainer'):
                    save_full_checkpoint_to_azure(
                        model.trainer,
                        connection_string,
                        CHECKPOINT_CONTAINER,
                        NUM_EPOCHS
                    )
                
            except Exception as e:
                print(f"Failed to upload final model: {e}")
        
        # Validate the model
        print("\n=== Running Final Validation ===")
        metrics = model.val()
        print(f"Box mAP50-95: {metrics.box.map}")
        print(f"Mask mAP50-95: {metrics.seg.map}")
    
    # Clean up resume weights if exists
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)
    
    print("\n✅ Training complete!")

if __name__ == '__main__':
    # Set multiprocessing start method
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    main()
