from ultralytics import YOLO
import torch
import argparse
import os
import json

def train_yolo(data_yaml, model_weights, epochs, batch_size, img_size, save_dir):
    model = YOLO(model_weights)

    results = model.train(
        data=data_yaml,  
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project=save_dir,  
        name=f"yolo_finetune_{os.path.basename(data_yaml).split('_')[2]}",  # Unique name based on ratio
        save=True,
        save_period=5,  
        val=True
    )

    history_path = os.path.join(save_dir, f"training_history_{os.path.basename(data_yaml).split('_')[2]}.json")
    with open(history_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Training completed! Results saved in {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a custom dataset with varying synthetic ratios")
    
    parser.add_argument("--synthetic_ratio", type=int, required=True, help="Synthetic data percentage (0, 5, 10, 15, 20)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--save_dir", type=str, default="runs/train", help="Directory to save results")

    args = parser.parse_args()

    # Construct the YAML file path based on the synthetic ratio
    data_yaml = f"~/dem-controlnet/Synthetic-Martian-Terrain-Image-Generation/datasets/new_datasets/crater_dataset_{args.synthetic_ratio}synthetic.yaml"

    train_yolo(data_yaml, args.weights, args.epochs, args.batch, args.imgsz, args.save_dir)