from ultralytics import YOLO

# Load a model  
model = YOLO("yolo11m.pt")

path = "/home/ubuntu/dem-controlnet/Synthetic-Martian-Terrain-Image-Generation/datasets/crater-detection/data.yaml"

# Train the model  
model.train(data=path, #path to yaml file  
           imgsz=640, #image size for training  
           batch=8, #number of batch size  
           epochs=100, #number of epochs  
           device=0) #device ‘0’ if gpu else ‘cpu’