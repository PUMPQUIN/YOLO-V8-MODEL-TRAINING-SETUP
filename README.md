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

#TEST CODE MAIN.PY FILE

    import os
    import cv2
    import numpy as np
    from ultralytics import YOLO

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    VIDEOS_DIR = os.path.join('.', 'videos')  # Path to videos folder
    video_path = os.path.join(VIDEOS_DIR, 'CHICKEN.mp4')  # Input video path
    video_path_out = f"{os.path.splitext(video_path)[0]}_out.mp4"  # Output video path
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')  # YOLO model path

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {os.path.abspath(video_path)}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {os.path.abspath(model_path)}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Failed to read the video file: {video_path}")

    H, W, _ = frame.shape
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    threshold = 0.1

    print("Processing video...")

    try:
        while ret:
            frame_copy = frame.copy()
            results = model(frame_copy, conf=threshold)

            for result in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = result
                if score > threshold and results[0].names[int(class_id)] == "chicken":
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    cv2.putText(frame_copy, "Chicken", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            out.write(frame_copy)
            cv2.imshow("Chicken Detection", frame_copy)  # Show output frame

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

            ret, frame = cap.read()

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Processing complete. Output saved to {video_path_out}")


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

 #CONVERTING BEST.PT FILE TO ONNX FORMAT FILE
 
     from ultralytics import YOLO
     model = YOLO("runs/detect/train/weights/best.pt") 
     model.export(format="onnx")

#TEST CODE USING PRE-TRAINED YOLOV8 FROM ULTRALYTICS(not recommended to use for final training!)

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt") 
    image_path = "C:/Users/FRANCIS QUINTIN JR/PycharmProjects/CHICK/pictures/CHICKEN2.jpg"
    results = model(image_path, conf=0.1)  
    for result in results:
        result.show()  


   
