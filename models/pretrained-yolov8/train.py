from ultralytics import YOLO
import torch
import yaml
import argparse
import os
import json

def train_yolo(data_yaml, model_weights, epochs, batch_size, img_size, save_dir):
    # Load YOLO model (fine-tuning from pretrained weights)
    model = YOLO(model_weights)

    # Train the model
    results = model.train(
        data=data_yaml,  # Path to dataset YAML
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project=save_dir,  # Where to save logs & weights
        name="yolo_finetune",
        save=True,
        save_period=5,  # Save every 5 epochs
        val=True
    )

    # Save training history (loss, metrics, etc.)
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Training completed! Results saved in {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a custom dataset")
    
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to dataset YAML file")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--save_dir", type=str, default="runs/train", help="Directory to save results")

    args = parser.parse_args()

    train_yolo(args.data, args.weights, args.epochs, args.batch, args.imgsz, args.save_dir)
