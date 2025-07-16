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
import albumentations as A

# Set YOLO config directory to avoid permission warnings
os.environ['YOLO_CONFIG_DIR'] = '/tmp/yolo_config'

# Suppress Ultralytics warnings during multiprocessing
warnings.filterwarnings('ignore', message='user config directory')
warnings.filterwarnings('ignore', message='Error decoding JSON')

def augment_dataset_5x(source_dir, output_dir, num_augmentations=4):
    """Create 5x dataset (original + 4 augmented versions)"""
    
    # Define augmentation pipeline
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.PiecewiseAffine(p=0.3),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.RandomScale(scale_limit=0.3, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Get all images
    img_paths = list(Path(source_dir).glob('**/*.jpg'))
    img_paths.extend(list(Path(source_dir).glob('**/*.png')))
    
    print(f"Found {len(img_paths)} images to augment")
    print(f"Creating 5x dataset in {output_dir}")
    
    # Process each image
    for img_path in tqdm(img_paths, desc="Augmenting images"):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        # Load corresponding label
        label_path = img_path.with_suffix('.txt')
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # YOLO format: class x y w h
                        class_labels.append(int(parts[0]))
                        bbox = [float(x) for x in parts[1:5]]
                        bboxes.append(bbox)
        
        # Prepare relative path for output
        rel_path = img_path.relative_to(source_dir)
        
        # Copy original (aug_0)
        out_img_path = Path(output_dir) / f"aug_0_{rel_path}"
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_path), img)
        
        # Copy original labels
        if label_path.exists():
            shutil.copy(label_path, out_img_path.with_suffix('.txt'))
        
        # Create augmented versions (aug_1 to aug_4)
        for i in range(1, num_augmentations + 1):
            # Apply augmentation
            try:
                augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                
                # Save augmented image
                out_img_path = Path(output_dir) / f"aug_{i}_{rel_path}"
                cv2.imwrite(str(out_img_path), augmented['image'])
                
                # Save augmented bboxes
                if augmented['bboxes']:
                    with open(out_img_path.with_suffix('.txt'), 'w') as f:
                        for bbox, class_label in zip(augmented['bboxes'], augmented['class_labels']):
                            # Write in YOLO format
                            f.write(f"{class_label} {' '.join(map(str, bbox))}\n")
                            
            except Exception as e:
                print(f"Warning: Augmentation failed for {img_path}: {e}")
                # If augmentation fails, copy original as fallback
                cv2.imwrite(str(out_img_path), img)
                if label_path.exists():
                    shutil.copy(label_path, out_img_path.with_suffix('.txt'))
    
    print(f"\n✅ Dataset augmentation complete!")
    print(f"Original images: {len(img_paths)}")
    print(f"Total images created: {len(img_paths) * 5}")


def integrate_with_azure_script(connection_string, container_name, local_dir):
    """Download data and create 5x augmented dataset"""
    
    # Original data directory
    original_dir = local_dir
    # Augmented data directory
    augmented_dir = local_dir + "_5x_augmented"
    
    # Check if augmented data already exists
    if os.path.exists(augmented_dir) and len(list(Path(augmented_dir).rglob('*.jpg'))) > 0:
        print(f"Augmented data already exists in '{augmented_dir}'. Using existing augmented data.")
        return augmented_dir
    
    # Augment the dataset
    print(f"\nCreating 5x augmented dataset...")
    augment_dataset_5x(original_dir, augmented_dir)
    
    # Update data.yaml for augmented dataset
    update_data_yaml_for_augmented(augmented_dir)
    
    return augmented_dir


def update_data_yaml_for_augmented(augmented_dir):
    """Update data.yaml to point to augmented directories"""
    data_yaml_path = os.path.join(augmented_dir, 'data.yaml')
    
    # Copy original data.yaml if it exists in parent
    original_yaml = os.path.join(os.path.dirname(augmented_dir), 'yolo_sidewalk', 'data.yaml')
    if os.path.exists(original_yaml) and not os.path.exists(data_yaml_path):
        shutil.copy(original_yaml, data_yaml_path)
    
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    else:
        data_config = {
            'nc': 1,
            'names': ['sidewalk']
        }
    
    # Update paths
    data_config['path'] = os.path.abspath(augmented_dir)
    data_config['train'] = os.path.join(os.path.abspath(augmented_dir), 'train/images')
    data_config['val'] = os.path.join(os.path.abspath(augmented_dir), 'val/images')
    if 'test' in data_config:
        data_config['test'] = os.path.join(os.path.abspath(augmented_dir), 'test/images')
    
    # Save updated config
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Updated data.yaml saved to {data_yaml_path}")

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
        print(f"\n🔄 Starting checkpoint save for epoch {epoch}...")

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

            # --- MODIFICATION START ---
            # Save the model weights by copying the last saved model from the trainer
            weights_path = os.path.join(checkpoint_dir, 'weights.pt')
            if trainer.last:  # Check if a path to the last model exists
                shutil.copy(trainer.last, weights_path)
            else: # Fallback for older versions or if trainer.last is not available
                trainer.model.save(weights_path)
            # --- MODIFICATION END ---
            
            print(f"  ✓ Model weights saved")

            # Save optimizer state
            optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
            torch.save(trainer.optimizer.state_dict(), optimizer_path)
            print(f"  ✓ Optimizer state saved")

            # ... (rest of the function remains the same) ...
            
            # Save training args and state
            training_state = {
                'epoch': epoch,
                'best_fitness': trainer.best_fitness,
                'fitness': trainer.fitness,
                'ema': trainer.ema.state_dict() if hasattr(trainer, 'ema') and trainer.ema else None,
                'updates': trainer.optimizer.state_dict()['state'].get(0, {})['step'] if trainer.optimizer and trainer.optimizer.state_dict()['state'] else 0,
                'train_args': vars(trainer.args),
            }

            state_path = os.path.join(checkpoint_dir, 'training_state.pt')
            torch.save(training_state, state_path)
            print(f"  ✓ Training state saved")

            # Save results CSV if exists
            if hasattr(trainer, 'csv') and trainer.csv and os.path.exists(trainer.csv):
                shutil.copy(trainer.csv, os.path.join(checkpoint_dir, 'results.csv'))
                print(f"  ✓ Results CSV saved")

            # Create a tar archive of the checkpoint directory
            tar_path = os.path.join(temp_dir, f'checkpoint_epoch_{epoch}.tar')
            shutil.make_archive(tar_path.replace('.tar', ''), 'tar', checkpoint_dir)
            print(f"  ✓ Archive created")

            # Upload the tar file to Azure
            blob_name = f'checkpoint_epoch_{epoch}.tar'
            blob_client = container_client.get_blob_client(blob_name)

            print(f"  📤 Uploading to Azure as: {blob_name}...")
            with open(tar_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"✅ Checkpoint for epoch {epoch} successfully uploaded to Azure!\n")

    except Exception as e:
        print(f"❌ WARNING: Failed to upload checkpoint to Azure. Error: {e}\n")
        import traceback
        traceback.print_exc()

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
            data_config['val'] = os.path.join(os.path.abspath(local_data_dir), 'val/images')
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
            'val': os.path.join(os.path.abspath(local_data_dir), 'val/images'),
            'test': os.path.join(os.path.abspath(local_data_dir), 'test/images'),
            'nc': 1,  # Number of classes
            'names': ['sidewalk']  # Class names
        }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        return data_yaml_path

def main():
    # --- Configuration ---
    SOURCE_DATA_CONTAINER = "13ksidewalk60"
    CHECKPOINT_CONTAINER = "13ksidewalk60yolotest"
    LOCAL_DATA_DIR = "/mnt/data/yolo_sidewalk"
    AUGMENTED_DATA_DIR = "/mnt/data/yolo_sidewalk_5x_augmented"  # New augmented data directory
    NUM_EPOCHS = 50
    BATCH_SIZE = 512  # Total batch size across all GPUs
    DEVICE = '0,1,2,3'  # Use 4 GPUs
    SAVE_PERIOD = 1

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

    # ===== NEW: CREATE 5X AUGMENTED DATASET =====
    # Check if augmented data already exists
    if not os.path.exists(AUGMENTED_DATA_DIR) or len(list(Path(AUGMENTED_DATA_DIR).rglob('*.jpg'))) == 0:
        print(f"\n🔄 Creating 5x augmented dataset...")
        print(f"This will create {5}x the original data through offline augmentation")
        augment_dataset_5x(LOCAL_DATA_DIR, AUGMENTED_DATA_DIR)
        print(f"✅ Augmentation complete! Data saved to {AUGMENTED_DATA_DIR}")
    else:
        print(f"\n✅ Using existing augmented data from {AUGMENTED_DATA_DIR}")
        print(f"Found {len(list(Path(AUGMENTED_DATA_DIR).rglob('*.jpg')))} images")
    
    # Use augmented data directory for training
    LOCAL_DATA_DIR = AUGMENTED_DATA_DIR
    # ===== END OF AUGMENTATION SECTION =====

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
                        model = YOLO('yolo11n-seg.pt')
                else:
                    print("Failed to extract checkpoint, starting from scratch...")
                    model = YOLO('yolo11n-seg.pt')
        else:
            print("No checkpoints found, starting from scratch...")
            model = YOLO('yolo11n-seg.pt')

    except Exception as e:
        print(f"Error checking for checkpoints: {e}")
        print("Starting training from scratch...")
        model = YOLO('yolo11n-seg.pt')

    # Store connection string and container name for callback
    callback_config = {
        'connection_string': connection_string,
        'container_name': CHECKPOINT_CONTAINER,
        'save_period': SAVE_PERIOD,
        'data_yaml': data_yaml_path,
        'start_epoch': start_epoch,
        'epoch_counter': 0  # Track epochs manually
    }

    # Custom training callback to save complete checkpoints
    def on_train_epoch_end(trainer):
        """Callback to save complete checkpoint and all metrics."""
        # Increment epoch counter
        callback_config['epoch_counter'] += 1
        actual_epoch = callback_config['start_epoch'] + callback_config['epoch_counter']
    
        print(f"\n--- Epoch {actual_epoch} Validation & Metrics ---")
        
        # Initialize a dictionary to hold all metrics for this epoch
        epoch_metrics = {
            'epoch': actual_epoch,
            'train_loss': {},
            'val_metrics': {}
        }
    
        try:
            # 1. Get training losses from the trainer object
            if hasattr(trainer, 'last_train_metrics'):
                epoch_metrics['train_loss'] = trainer.last_train_metrics
    
            # 2. Run validation and get validation metrics
            metrics = trainer.model.val(data=callback_config['data_yaml'], verbose=False)
            
            # Store box and segmentation metrics
            epoch_metrics['val_metrics'] = {
                'box': {
                    'precision': metrics.box.p.mean(),
                    'recall': metrics.box.r.mean(),
                    'map50': metrics.box.map50,
                    'map50_95': metrics.box.map
                },
                'seg': {
                    'precision': metrics.seg.p.mean(),
                    'recall': metrics.seg.r.mean(),
                    'map50': metrics.seg.map50,
                    'map50_95': metrics.seg.map
                }
            }
    
            # Print metrics for real-time monitoring
            print(f"  Training Box Loss: {epoch_metrics['train_loss'].get('box_loss', 'N/A'):.4f}")
            print(f"  Training Seg Loss: {epoch_metrics['train_loss'].get('seg_loss', 'N/A'):.4f}")
            print(f"  Validation mAP50-95 (Box): {epoch_metrics['val_metrics']['box']['map50_95']:.4f}")
            print(f"  Validation mAP50-95 (Seg): {epoch_metrics['val_metrics']['seg']['map50_95']:.4f}")
    
        except Exception as e:
            print(f"Error calculating or retrieving metrics: {e}")
        
        print("---------------------------------------------")
    
        # 3. Save the collected metrics to a file in /mnt/data
        results_path = "/mnt/data/training_metrics.jsonl"
        try:
            with open(results_path, 'a') as f:
                # Convert dictionary to a JSON string and write it as a new line
                f.write(json.dumps(epoch_metrics) + '\n')
            print(f"✅ Metrics for epoch {actual_epoch} saved to {results_path}")
        except Exception as e:
            print(f"❌ Failed to save metrics for epoch {actual_epoch}. Error: {e}")
    
        # --- Your existing checkpoint saving logic ---
        if actual_epoch % callback_config['save_period'] == 0:
            print(f"📸 Saving checkpoint at epoch {actual_epoch}...")
            save_full_checkpoint_to_azure(
                trainer,
                callback_config['connection_string'],
                callback_config['container_name'],
                actual_epoch
            )
        else:
            next_save = ((actual_epoch // callback_config['save_period']) + 1) * callback_config['save_period']
            print(f"ℹ️  Epoch {actual_epoch} - Not saving checkpoint (next save at epoch {next_save})")

    # Also add a callback that runs after each epoch completes (alternative hook)
    def on_train_epoch_end_alt(trainer):
        """Alternative callback using on_epoch_end hook."""
        # This might work better for checkpoint saving
        callback_config['epoch_counter'] = trainer.epoch + 1
        actual_epoch = callback_config['start_epoch'] + callback_config['epoch_counter']

        if actual_epoch % callback_config['save_period'] == 0:
            print(f"📸 [Alt] Saving checkpoint at epoch {actual_epoch}")
            save_full_checkpoint_to_azure(
                trainer,
                callback_config['connection_string'],
                callback_config['container_name'],
                actual_epoch
            )

    # Add callbacks to model
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    # Try both hooks to ensure we catch the epoch end
    model.add_callback("on_epoch_end", on_train_epoch_end_alt)

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
        print(f"📊 Using 5x augmented dataset with {len(list(Path(LOCAL_DATA_DIR).rglob('*.jpg')))} total images")

        # Prepare training arguments
        train_args = {
            'data': data_yaml_path,
            'epochs': remaining_epochs,  # Train only remaining epochs
            'batch': BATCH_SIZE,
            'imgsz': 640,
            'device': DEVICE,
            'workers': 8,
            'patience': 30,  # Disable early stopping
            'save': True,
            'save_period': -1,  # Disable default saving
            'project': temp_project_dir,
            'name': 'yolov11n_seg_sidewalk',
            'exist_ok': True,
            'pretrained': False,  # Already have a model
            'resume': resume_path if resume_path and os.path.exists(resume_path) else False,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': True,
            'amp': True,

            # ===== REDUCED AUGMENTATION PARAMETERS (since data is pre-augmented) =====
            'hsv_h': 0.015,   # Keep minimal color variation
            'hsv_s': 0.4,     # Reduced from 0.7
            'hsv_v': 0.2,     # Reduced from 0.4
            'degrees': 0.0,   # No rotation (already in augmented data)
            'translate': 0.05, # Minimal translation
            'scale': 0.2,     # Reduced from 0.5
            'shear': 0.0,     # No shear (already in augmented data)
            'perspective': 0.0, # No perspective (already in augmented data)
            'flipud': 0.0,    # No vertical flip (already in augmented data)
            'fliplr': 0.5,    # Keep horizontal flip for additional variation
            'mosaic': 0.5,    # Reduced mosaic probability
            'mixup': 0.0,     # No mixup needed
            'copy_paste': 0.0, # No copy-paste needed
            # ===== END OF AUGMENTATION PARAMETERS =====

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
        final_blob_name = "yolov11n_seg_sidewalk_final.pt"
        # Point to the model from the last epoch instead of the best one
        final_model_path = os.path.join(temp_project_dir, 'yolov11n_seg_sidewalk', 'weights', 'last.pt')
        
        if os.path.exists(final_model_path):
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                blob_client = blob_service_client.get_blob_client(
                    container=CHECKPOINT_CONTAINER, 
                    blob=final_blob_name
                )
        
                with open(final_model_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"✅ Final model from the last epoch uploaded to Azure as: {final_blob_name}")
        
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
    print(f"📊 Trained on 5x augmented dataset ({len(list(Path(LOCAL_DATA_DIR).rglob('*.jpg')))} total images)")
    
if __name__ == '__main__':
    # Set multiprocessing start method
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    main()
