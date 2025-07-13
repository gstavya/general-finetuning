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
from datetime import datetime
import threading
from flask import Flask, render_template_string, jsonify
import logging

# Set YOLO config directory to avoid permission warnings
os.environ['YOLO_CONFIG_DIR'] = '/tmp/yolo_config'

# Suppress Ultralytics warnings during multiprocessing
warnings.filterwarnings('ignore', message='user config directory')
warnings.filterwarnings('ignore', message='Error decoding JSON')

# Suppress Flask development server warning
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class MetricsCollector:
    """Collects and manages training metrics for visualization."""
    
    def __init__(self, metrics_file='metrics.json', connection_string=None, container_name=None):
        self.metrics_file = metrics_file
        self.connection_string = connection_string
        self.container_name = container_name
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'box_precision': [],
            'box_recall': [],
            'box_map50': [],
            'box_map': [],
            'mask_precision': [],
            'mask_recall': [],
            'mask_map50': [],
            'mask_map': [],
            'learning_rate': [],
            'time_per_epoch': [],
            'timestamps': [],
            'start_time': datetime.now().isoformat(),
            'total_epochs': 0,
            'current_epoch': 0
        }
        self.epoch_start_time = None
        self.load_existing_metrics()
    
    def load_existing_metrics(self):
        """Load existing metrics from file if available."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                print(f"Loaded existing metrics from {self.metrics_file}")
            except Exception as e:
                print(f"Could not load existing metrics: {e}")
    
    def start_epoch(self):
        """Mark the start of a new epoch."""
        self.epoch_start_time = time.time()
    
    def add_epoch_metrics(self, epoch, trainer, validation_metrics=None):
        """Add metrics for the current epoch."""
        try:
            # Calculate time for this epoch
            epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
            
            # Update basic info
            self.metrics['current_epoch'] = epoch
            self.metrics['epochs'].append(epoch)
            self.metrics['time_per_epoch'].append(epoch_time)
            self.metrics['timestamps'].append(datetime.now().isoformat())
            
            # Get training loss (if available)
            if hasattr(trainer, 'loss'):
                self.metrics['train_loss'].append(float(trainer.loss))
            else:
                self.metrics['train_loss'].append(None)
            
            # Get learning rate
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                lr = trainer.optimizer.param_groups[0]['lr']
                self.metrics['learning_rate'].append(float(lr))
            else:
                self.metrics['learning_rate'].append(None)
            
            # Add validation metrics if provided
            if validation_metrics:
                self.metrics['val_loss'].append(float(validation_metrics.box.fitness) if hasattr(validation_metrics.box, 'fitness') else None)
                self.metrics['box_precision'].append(float(validation_metrics.box.p.mean()))
                self.metrics['box_recall'].append(float(validation_metrics.box.r.mean()))
                self.metrics['box_map50'].append(float(validation_metrics.box.map50))
                self.metrics['box_map'].append(float(validation_metrics.box.map))
                self.metrics['mask_precision'].append(float(validation_metrics.seg.p.mean()))
                self.metrics['mask_recall'].append(float(validation_metrics.seg.r.mean()))
                self.metrics['mask_map50'].append(float(validation_metrics.seg.map50))
                self.metrics['mask_map'].append(float(validation_metrics.seg.map))
            else:
                # Append None if no validation metrics
                for key in ['val_loss', 'box_precision', 'box_recall', 'box_map50', 'box_map',
                           'mask_precision', 'mask_recall', 'mask_map50', 'mask_map']:
                    self.metrics[key].append(None)
            
            # Save to file
            self.save_metrics()
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
    
    def save_metrics(self):
        """Save metrics to JSON file and optionally to Azure."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            # Upload to Azure if configured
            if self.connection_string and self.container_name:
                self.upload_metrics_to_azure()
                
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def upload_metrics_to_azure(self):
        """Upload metrics JSON to Azure."""
        try:
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            blob_client = blob_service_client.get_blob_client(
                container=self.container_name, 
                blob='training_metrics.json'
            )
            
            with open(self.metrics_file, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
                
        except Exception as e:
            print(f"Failed to upload metrics to Azure: {e}")
    
    def set_total_epochs(self, total_epochs):
        """Set the total number of epochs for training."""
        self.metrics['total_epochs'] = total_epochs


# HTML template for the dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .info-bar {
            background-color: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        .info-item {
            text-align: center;
            margin: 10px;
        }
        .info-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
        .info-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .chart-container {
            background-color: white;
            margin-bottom: 20px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart {
            width: 100%;
            height: 400px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }
        .status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
        }
        .status.connected {
            background-color: #4CAF50;
            color: white;
        }
        .status.disconnected {
            background-color: #f44336;
            color: white;
        }
    </style>
</head>
<body>
    <div class="status connected" id="status">Connected</div>
    <h1>YOLO Training Dashboard</h1>
    
    <div class="info-bar" id="info-bar">
        <div class="info-item">
            <div class="info-label">Current Epoch</div>
            <div class="info-value" id="current-epoch">-</div>
        </div>
        <div class="info-item">
            <div class="info-label">Total Epochs</div>
            <div class="info-value" id="total-epochs">-</div>
        </div>
        <div class="info-item">
            <div class="info-label">Best mAP50-95</div>
            <div class="info-value" id="best-map">-</div>
        </div>
        <div class="info-item">
            <div class="info-label">Time per Epoch</div>
            <div class="info-value" id="time-per-epoch">-</div>
        </div>
        <div class="info-item">
            <div class="info-label">ETA</div>
            <div class="info-value" id="eta">-</div>
        </div>
    </div>
    
    <div class="grid">
        <div class="chart-container">
            <div id="loss-chart" class="chart"></div>
        </div>
        <div class="chart-container">
            <div id="map-chart" class="chart"></div>
        </div>
        <div class="chart-container">
            <div id="precision-recall-chart" class="chart"></div>
        </div>
        <div class="chart-container">
            <div id="learning-rate-chart" class="chart"></div>
        </div>
    </div>
    
    <script>
        let lastUpdate = null;
        
        function updateDashboard() {
            fetch('/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update info bar
                    document.getElementById('current-epoch').textContent = data.current_epoch || '-';
                    document.getElementById('total-epochs').textContent = data.total_epochs || '-';
                    
                    // Calculate best mAP
                    const boxMaps = data.box_map.filter(v => v !== null);
                    const bestMap = boxMaps.length > 0 ? Math.max(...boxMaps).toFixed(4) : '-';
                    document.getElementById('best-map').textContent = bestMap;
                    
                    // Calculate average time per epoch
                    const times = data.time_per_epoch.filter(v => v !== null);
                    const avgTime = times.length > 0 ? (times.reduce((a, b) => a + b, 0) / times.length).toFixed(1) : '-';
                    document.getElementById('time-per-epoch').textContent = avgTime + 's';
                    
                    // Calculate ETA
                    if (data.current_epoch && data.total_epochs && avgTime !== '-') {
                        const remainingEpochs = data.total_epochs - data.current_epoch;
                        const etaSeconds = remainingEpochs * parseFloat(avgTime);
                        const etaHours = Math.floor(etaSeconds / 3600);
                        const etaMinutes = Math.floor((etaSeconds % 3600) / 60);
                        document.getElementById('eta').textContent = `${etaHours}h ${etaMinutes}m`;
                    } else {
                        document.getElementById('eta').textContent = '-';
                    }
                    
                    // Update charts
                    updateCharts(data);
                    
                    // Update status
                    document.getElementById('status').className = 'status connected';
                    document.getElementById('status').textContent = 'Connected';
                    lastUpdate = Date.now();
                })
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                    document.getElementById('status').className = 'status disconnected';
                    document.getElementById('status').textContent = 'Disconnected';
                });
        }
        
        function updateCharts(data) {
            // Loss Chart
            const lossTraces = [
                {
                    x: data.epochs,
                    y: data.train_loss,
                    name: 'Train Loss',
                    type: 'scatter',
                    mode: 'lines+markers'
                }
            ];
            
            if (data.val_loss.some(v => v !== null)) {
                lossTraces.push({
                    x: data.epochs,
                    y: data.val_loss,
                    name: 'Val Loss',
                    type: 'scatter',
                    mode: 'lines+markers'
                });
            }
            
            Plotly.newPlot('loss-chart', lossTraces, {
                title: 'Training and Validation Loss',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Loss' }
            });
            
            // mAP Chart
            const mapTraces = [
                {
                    x: data.epochs,
                    y: data.box_map50,
                    name: 'Box mAP50',
                    type: 'scatter',
                    mode: 'lines+markers'
                },
                {
                    x: data.epochs,
                    y: data.box_map,
                    name: 'Box mAP50-95',
                    type: 'scatter',
                    mode: 'lines+markers'
                },
                {
                    x: data.epochs,
                    y: data.mask_map50,
                    name: 'Mask mAP50',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { dash: 'dash' }
                },
                {
                    x: data.epochs,
                    y: data.mask_map,
                    name: 'Mask mAP50-95',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { dash: 'dash' }
                }
            ];
            
            Plotly.newPlot('map-chart', mapTraces, {
                title: 'Mean Average Precision (mAP)',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'mAP', range: [0, 1] }
            });
            
            // Precision-Recall Chart
            const prTraces = [
                {
                    x: data.epochs,
                    y: data.box_precision,
                    name: 'Box Precision',
                    type: 'scatter',
                    mode: 'lines+markers'
                },
                {
                    x: data.epochs,
                    y: data.box_recall,
                    name: 'Box Recall',
                    type: 'scatter',
                    mode: 'lines+markers'
                },
                {
                    x: data.epochs,
                    y: data.mask_precision,
                    name: 'Mask Precision',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { dash: 'dash' }
                },
                {
                    x: data.epochs,
                    y: data.mask_recall,
                    name: 'Mask Recall',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { dash: 'dash' }
                }
            ];
            
            Plotly.newPlot('precision-recall-chart', prTraces, {
                title: 'Precision and Recall',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Score', range: [0, 1] }
            });
            
            // Learning Rate Chart
            const lrTrace = [{
                x: data.epochs,
                y: data.learning_rate,
                name: 'Learning Rate',
                type: 'scatter',
                mode: 'lines+markers'
            }];
            
            Plotly.newPlot('learning-rate-chart', lrTrace, {
                title: 'Learning Rate Schedule',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Learning Rate', type: 'log' }
            });
        }
        
        // Initial update
        updateDashboard();
        
        // Update every 5 seconds
        setInterval(updateDashboard, 5000);
        
        // Check connection status
        setInterval(() => {
            if (lastUpdate && Date.now() - lastUpdate > 15000) {
                document.getElementById('status').className = 'status disconnected';
                document.getElementById('status').textContent = 'Disconnected';
            }
        }, 1000);
    </script>
</body>
</html>
'''


def run_visualization_server(metrics_collector, port=5000):
    """Run the Flask visualization server."""
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/metrics')
    def get_metrics():
        return jsonify(metrics_collector.metrics)
    
    # Run in a separate thread
    def run_server():
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print(f"\n📊 Visualization dashboard running at http://localhost:{port}\n")


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
            
            # Save the model weights
            weights_path = os.path.join(checkpoint_dir, 'weights.pt')
            trainer.save_model(weights_path)
            print(f"  ✓ Model weights saved")
            
            # Save optimizer state
            optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
            torch.save(trainer.optimizer.state_dict(), optimizer_path)
            print(f"  ✓ Optimizer state saved")
            
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
    SOURCE_DATA_CONTAINER = "2ktestsidewalk60"
    CHECKPOINT_CONTAINER = "quicktestyolo"
    LOCAL_DATA_DIR = "/mnt/data/yolo_sidewalk"
    NUM_EPOCHS = 2
    BATCH_SIZE = 64  # Total batch size across all GPUs
    DEVICE = '0,1,2,3'  # Use 4 GPUs
    SAVE_PERIOD = 5
    VISUALIZATION_PORT = 5000
    
    # Create YOLO config directory
    os.makedirs('/tmp/yolo_config', exist_ok=True)
    
    # Azure connection string
    connection_string = "DefaultEndpointsProtocol=https;AccountName=resnettrainingdata;AccountKey=afq0lgt0sj3lq1+b3Y6eeIg+JArkqE5UJL7tHSeM+Bxa0S3aQSK9ZRMZHozG1PJx2rGfwBh7DySr+ASt3w6JmA==;EndpointSuffix=core.windows.net"
    
    print(f"Using devices: {DEVICE}")
    print(f"Total batch size: {BATCH_SIZE} (will be split across 4 GPUs)")
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector(
        metrics_file='training_metrics.json',
        connection_string=connection_string,
        container_name=CHECKPOINT_CONTAINER
    )
    
    # Start visualization server
    run_visualization_server(metrics_collector, port=VISUALIZATION_PORT)
    
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
                        model = YOLO('yolo11x-seg.pt')
                else:
                    print("Failed to extract checkpoint, starting from scratch...")
                    model = YOLO('yolo11x-seg.pt')
        else:
            print("No checkpoints found, starting from scratch...")
            model = YOLO('yolo11x-seg.pt')
    
    except Exception as e:
        print(f"Error checking for checkpoints: {e}")
        print("Starting training from scratch...")
        model = YOLO('yolo11x-seg.pt')
    
    # Set total epochs in metrics collector
    metrics_collector.set_total_epochs(NUM_EPOCHS)
    
    # Store connection string and container name for callback
    callback_config = {
        'connection_string': connection_string,
        'container_name': CHECKPOINT_CONTAINER,
        'save_period': SAVE_PERIOD,
        'data_yaml': data_yaml_path,
        'start_epoch': start_epoch,
        'epoch_counter': 0,  # Track epochs manually
        'metrics_collector': metrics_collector
    }
    
    # Custom training callback to save complete checkpoints
    def on_train_epoch_end(trainer):
        """Callback to save complete checkpoint and display metrics."""
        # Start timing for next epoch
        callback_config['metrics_collector'].start_epoch()
        
        # Increment epoch counter
        callback_config['epoch_counter'] += 1
        
        # Calculate actual epoch considering resume
        actual_epoch = callback_config['start_epoch'] + callback_config['epoch_counter']
        
        # Display validation metrics
        print(f"\n--- Epoch {actual_epoch} Validation Metrics ---")
        print(f"(Internal epoch: {trainer.epoch + 1}, Counter: {callback_config['epoch_counter']})")
        
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
            
            # Add metrics to collector
            callback_config['metrics_collector'].add_epoch_metrics(actual_epoch, trainer, metrics)
            
        except Exception as e:
            print(f"Error calculating validation metrics: {e}")
            print("------------------------------------\n")
            # Still add metrics even if validation fails
            callback_config['metrics_collector'].add_epoch_metrics(actual_epoch, trainer, None)
        
        # Save checkpoint every SAVE_PERIOD epochs
        if actual_epoch % callback_config['save_period'] == 0:
            print(f"📸 Saving checkpoint at epoch {actual_epoch} (divisible by {callback_config['save_period']})")
            save_full_checkpoint_to_azure(
                trainer,
                callback_config['connection_string'],
                callback_config['container_name'],
                actual_epoch
            )
        else:
            print(f"ℹ️  Epoch {actual_epoch} - Not saving (next save at epoch {((actual_epoch // callback_config['save_period']) + 1) * callback_config['save_period']})")
    
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
    
    # Callback to start epoch timer
    def on_train_epoch_start(trainer):
        """Callback to start timing the epoch."""
        callback_config['metrics_collector'].start_epoch()
    
    # Add callbacks to model
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
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
            'name': 'yolov11x_seg_sidewalk',
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
        final_blob_name = "yolov11x_seg_sidewalk_final.pt"
        best_model_path = os.path.join(temp_project_dir, 'yolov11x_seg_sidewalk', 'weights', 'best.pt')
        
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
        
        # Save final metrics
        metrics_collector.save_metrics()
    
    # Clean up resume weights if exists
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)
    
    print("\n✅ Training complete!")
    print(f"📊 Metrics saved to: training_metrics.json")
    print(f"📊 Dashboard was available at: http://localhost:{VISUALIZATION_PORT}")

if __name__ == '__main__':
    # Set multiprocessing start method
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    main()
