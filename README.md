# YOLO-V8-MODEL-TRAINING-SETUP
    import torch
    import os
    from ultralytics import YOLO

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = True

    if __name__ == '__main__':
  
        model = YOLO("yolov8n.pt")
   
        results = model.train(
            data="config.yaml",  # Path to dataset configuration
            epochs=100,          # Number of training epochs
            imgsz=640,           # Higher resolution for better accuracy
            batch=4,             # Lower batch size to fit into 4GB GPU
            workers=0,           # Avoid multiprocessing issues on Windows
            amp=True             # Use Automatic Mixed Precision (reduces memory usage)
        )

        model.export(format="onnx")

        print("✅ Training completed with optimized settings!")

#CONFIG.YAML FILE

    path: C:/Users/FRANCIS QUINTIN JR/PycharmProjects/CHICK/data
    train: train/images
    val: valid/images

    nc: 1
    names: ['chicken']
    roboflow:
      workspace: pumpquin
      project: chicken-segmentation-byru8-j3yvi
      version: 1
      license: CC BY 4.0
      url: https://universe.roboflow.com/pumpquin/chicken-segmentation-byru8-j3yvi/dataset/1
