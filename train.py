import os
import yaml
import argparse
from ultralytics import YOLO

def train_model(data_yaml_path, model_variant='yolov8n.pt', epochs=100, imgsz=640, batch=16):
    """
    Trains a YOLO model based on the provided dataset.
    """
    # 1. Load the data.yaml to verify/fix paths
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Ensure paths in data.yaml are absolute to avoid confusion
    dataset_root = os.path.dirname(os.path.abspath(data_yaml_path))
    
    # Update paths if they are relative
    for key in ['train', 'val', 'test']:
        if key in data:
            path = data[key]
            if not os.path.isabs(path):
                # If it starts with ../, we might need to handle it. 
                # The provided yaml had ../train/images which seems wrong for the structure.
                # We'll normalize it relative to the yaml file.
                full_path = os.path.abspath(os.path.join(dataset_root, path))
                data[key] = full_path
    
    # Save a temporary fixed yaml
    tmp_yaml = 'temp_data.yaml'
    with open(tmp_yaml, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Using fixed dataset config: {data}")

    # 2. Load a model
    model = YOLO(model_variant)  # load a pretrained model (recommended for training)

    # 3. Train the model
    results = model.train(
        data=tmp_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='weed_wacker_model'
    )
    
    print(f"Training complete. Results saved to {results.save_dir}")
    
    # Clean up temp yaml
    if os.path.exists(tmp_yaml):
        os.remove(tmp_yaml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for Weed Detection")
    parser.add_argument("--data", type=str, default="DATASETS/WeedCrop.v1i.yolov5pytorch/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model variant (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    # Check if ultralytics is installed, if not, suggest installation
    try:
        import ultralytics
    except ImportError:
        print("Ultralytics not found. Please install it using: pip install ultralytics")
        exit(1)

    train_model(
        data_yaml_path=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
