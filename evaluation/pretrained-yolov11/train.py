from ultralytics import YOLO
import torch
import argparse
import os
import json

def train_yolo(data_yaml, model_weights, epochs, batch_size, img_size, save_dir):
    # Check if the provided data YAML file exists
    if not os.path.exists(data_yaml):
        print(f"Error: The specified YAML file {data_yaml} does not exist!")
        return

    # Initialize YOLO model
    model = YOLO(model_weights)

    # Start training process
    results = model.train(
        data=data_yaml,  
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project=save_dir,  
        name=f"yolo_finetune_{os.path.basename(data_yaml)}",  # Unique name based on ratio
        save=True,
        save_period=5,  
        val=True
    )

    # Extract only the serializable parts of the results (assuming results is a dictionary)
    # If DetMetrics or other objects are in results, we need to extract relevant data.
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = value
        elif isinstance(value, list):
            serializable_results[key] = value
        else:
            # Convert non-serializable objects (like DetMetrics) into strings or simple data types
            serializable_results[key] = str(value)  # or any other way to extract meaningful data

    # Save training history
    history_path = os.path.join(save_dir, f"training_history_{os.path.basename(data_yaml)}.json")
    with open(history_path, "w") as f:
        json.dump(serializable_results, f, indent=4)

    print(f"Training completed! Results saved in {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 on a custom dataset with varying synthetic ratios")
    
    # Add argument for synthetic ratio with validation to ensure it is a valid value
    parser.add_argument("--synthetic_ratio", type=int, required=True, choices=[0, 10, 20],
                        help="Synthetic data percentage (0, 10, 20)")

    parser.add_argument("--weights", type=str, default="yolov11n.pt", help="Pretrained model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--save_dir", type=str, default="runs/train", help="Directory to save results")

    args = parser.parse_args()

    # Construct the YAML file path based on the synthetic ratio
    data_yaml = f"/home/ubuntu/dem-controlnet/Synthetic-Martian-Terrain-Image-Generation/datasets/Dataset_special_{args.synthetic_ratio}/data.yaml"



    # Call the train_yolo function
    train_yolo(data_yaml, args.weights, args.epochs, args.batch, args.imgsz, args.save_dir)
