from ultralytics import YOLO

def load_model():
    model = YOLO("yolo11n.pt")
    return model

def train(model, path, img_size, batch_size, epochs, device='cpu'):
    model.train(data=path, imgsz=img_size,batch=batch_size,epochs=epochs,device=device)
    return model

if __name__ == "__main__":
    model = load_model()

    # Ratio 0.05
    path = "yolo_datasets/syn0.05_size1000_90-5-5/data.yaml"
    img_size=640
    batch_size = 8
    epochs = 200
    device='cpu'
    model = train(model, path, img_size, batch_size, epochs, device)

    # # Ratio 0.50
    # path = "yolo_datasets/syn0.5_size1000_90-5-5/data.yaml"
    # img_size=640
    # batch_size = 8
    # epochs = 200
    # device='cpu'
    # model = train(model, path, img_size, batch_size, epochs, device)