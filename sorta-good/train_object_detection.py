from ultralytics import YOLO
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_history(csv_path, save_dir):
    """
    Reads the results.csv from YOLO training and plots loss curves.
    """
    try:
        # Read CSV, stripping whitespace from column names
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # Create a figure with subplots for Box, Class, and DFL loss
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        
        losses = ['box_loss', 'cls_loss', 'dfl_loss']
        titles = ['Box Loss', 'Class Loss', 'DFL Loss']
        
        for i, loss_name in enumerate(losses):
            train_col = f'train/{loss_name}'
            val_col = f'val/{loss_name}'
            
            if train_col in df.columns and val_col in df.columns:
                ax[i].plot(df['epoch'], df[train_col], label='Train')
                ax[i].plot(df['epoch'], df[val_col], label='Val')
                ax[i].set_title(titles[i])
                ax[i].set_xlabel('Epoch')
                ax[i].set_ylabel('Loss')
                ax[i].legend()
                ax[i].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'loss_minimization.png')
        plt.savefig(save_path)
        print(f"Custom loss graphs saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot training history. {e}")

def train_custom_model(dataset_dir):
    """
    Trains a YOLOv8 model on a custom dataset.
    """
    # 1. Load a model
    # 'yolov8n.pt' is the Nano model (fastest). 
    # Use 'yolov8m.pt' or 'yolov8x.pt' for higher accuracy but slower speed.
    # The model weights will automatically download if not present.
    print("Loading pre-trained YOLOv8 model...")
    model = YOLO('yolov8x.pt') 

    # 2. Define path to your data configuration file
    # Ensure this points to the actual location of your data.yaml
    yaml_path = os.path.join(dataset_dir, 'data.yaml')

    if not os.path.exists(yaml_path):
        print(f"Error: Configuration file not found at {yaml_path}")
        print("Please ensure you have created the data.yaml file describing your dataset.")
        return

    # 3. Train the model
    # epochs: Number of full passes through the dataset
    # imgsz: Image size (pixels). 640 is standard.
    # device: 0 for GPU, 'cpu' for CPU. (Auto-detects if not specified)
    run_name = os.path.basename(os.path.normpath(dataset_dir))

    print("Starting training...")
    results = model.train(
        data=yaml_path,
        epochs=300,       # Increased for small dataset convergence
        imgsz=640,
        patience=75,      # Stop training earlier if no improvement is observed 
        batch=16,         # Adjust based on your GPU memory
        name=run_name,       # Name of the run (saved in runs/detect/)
        exist_ok=True,       # Overwrite existing experiment
        freeze=10,           # Freeze backbone layers to prevent overfitting on small data
        device='cuda:0'
    )

    # 4. Validate the model
    print("Validating model on validation set...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

    # 5. Plot training graphs
    if hasattr(model, 'trainer') and model.trainer.save_dir:
        save_dir = str(model.trainer.save_dir)
        csv_path = os.path.join(save_dir, 'results.csv')
        if os.path.exists(csv_path):
            plot_training_history(csv_path, save_dir)

    # 6. Export the model (Optional)
    # Useful for deployment (e.g., to ONNX for generic use)
    model.export(format='onnx')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv8 on custom dataset")
    parser.add_argument("--dataset_dir", type=str, default="datasets/dataset_pumpkin", 
                        help="Path to the dataset directory containing data.yaml")
    args = parser.parse_args()
    # Entry point is required for multiprocessing on Windows/Linux
    train_custom_model(args.dataset_dir)