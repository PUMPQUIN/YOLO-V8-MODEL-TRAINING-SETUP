# CHECKING IF YOU HAVE GPU TO YOUR PYTHON SCRIPT ENVIRONMENT USING PYCHARM

    import torch

    print("✅ CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("🖥️ GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ No GPU detected. Training will use CPU only.")


# YOLO-V8-MODEL-TRAINING-SETUP
    import torch
    import os
    from ultralytics import YOLO

    # Set PyTorch memory config to prevent OOM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Clear cache and optimize cuDNN backend
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    if __name__ == '__main__':
        # Load YOLOv8 nano model (you can also try 'yolov8s.pt' for more accuracy)
        model = YOLO("yolov8n.pt")

        # Train model on your custom chicken dataset
        results = model.train(
            data="config.yaml",   # Path to your dataset config file
            epochs=100,           # Number of epochs (you can lower for testing)
            imgsz=640,            # Input image size
            batch=4,              # Smaller batch for limited VRAM (4GB~6GB GPUs)
            workers=0,            # Disable multiprocessing (avoid issues on Windows)
            amp=True              # Automatic Mixed Precision (faster & less memory)
        )

        # Export model in PyTorch (.pt) format
        model.export(format="pt")  # You will get a 'best.pt' file

        print("✅ Training complete. Model exported in PyTorch format!")


#TEST CODE MAIN.PY FILE

    import os
    import cv2
    from ultralytics import YOLO

    # Allow some libraries to avoid OpenMP issues
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Define paths
    VIDEOS_DIR = os.path.join('.', 'videos')
    video_path = os.path.join(VIDEOS_DIR, 'DUSK #2.mp4')  # Your input video
    video_path_out = f"{os.path.splitext(video_path)[0]}_out.mp4"
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')  # Trained YOLOv8 model

    # Check paths
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {os.path.abspath(video_path)}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {os.path.abspath(model_path)}")

    # Load YOLOv8 model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Failed to read the video file: {video_path}")

    # Get frame size and fps
    H, W, _ = frame.shape
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    threshold = 0.5  # Confidence threshold

    print("🔄 Processing video... Press 'q' to quit.")

    try:
        while ret:
            frame_copy = frame.copy()

            # Run YOLOv8 inference
            results = model(frame_copy, conf=threshold)

            # Draw detections
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = float(box.conf[0])               # Confidence score
                class_id = int(box.cls[0])              # Class ID
                label = results[0].names[class_id]      # Class label

                if conf > threshold and label == "CHICKEN":
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_copy, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Write and show
            out.write(frame_copy)
            cv2.imshow("POULTRIYA", frame_copy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"✅ Processing complete. Output saved to: {video_path_out}")


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

#BYTETRACK.YAML FILE

    tracker_type: bytetrack
    track_high_thresh: 0.05
    track_low_thresh: 0.05
    new_track_thresh: 0.05
    track_buffer: 120
    match_thresh: 0.3
    frame_rate: 15
    mot20: false
    fuse_score: false  # Required to prevent AttributeError in Ultralytics



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


#TEST CODE USING ESP32 CAM CHICKEN COUNTER
   
    import requests
    import cv2
    import numpy as np
    import re
    from ultralytics import YOLO

    # Load YOLOv8 model
    model_path = "runs/detect/train/weights/best.pt"
    model = YOLO(model_path)

    threshold = 0.5
    stream_url = "http://192.168.200.14/stream"

    print("📡 Connecting to ESP32-CAM stream...")

    # Start the MJPEG stream
    stream = requests.get(stream_url, stream=True)
    if stream.status_code != 200:
        raise RuntimeError(f"❌ Failed to connect. Status code: {stream.status_code}")

    # MJPEG frame pattern (JPEG start/end)
    bytes_data = b""
    pattern = re.compile(b'\xff\xd8.*?\xff\xd9', re.DOTALL)

    print("✅ Connected. Running YOLO detection. Press 'q' to quit.")

    try:
        for chunk in stream.iter_content(chunk_size=4096):
            bytes_data += chunk

            # Search for complete JPEG frames
            matches = list(pattern.finditer(bytes_data))
            for match in matches:
                jpg = match.group()
                bytes_data = bytes_data[match.end():]  # discard used bytes

                try:
                    img_array = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                except Exception as e:
                    print(f"⚠️ JPEG decode error: {e}")
                    continue

                # Run YOLOv8 detection
                results = model(frame, conf=threshold)

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = results[0].names[class_id]

                    if label == "CHICKEN":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("ESP32-CAM YOLO Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    except Exception as e:
        print(f"❌ Runtime error: {e}")
    finally:
        stream.close()
        cv2.destroyAllWindows()
        print("✅ Stream closed and resources released.")


#TEST CODE USING CCTV CHICKEN COUNTER

    import os
    import cv2
    import torch
    import time
    import zmq
    import numpy as np
    import threading
    from ultralytics import YOLO
    from collections import defaultdict
    from queue import Queue, Empty

    # ----------------------- Configuration ----------------------- #

    RTSP_URL = "rtsp://PUMPQUIN:030909_FQ@192.168.200.57:554/stream2"
    MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
    TRACKER_CONFIG = "bytetrack.yaml"
    ZMQ_PORT = 5555
    TARGET_WIDTH, TARGET_HEIGHT = 640, 380
    COUNTING_LINE_X = 320
    COUNTING_LINE_Y1, COUNTING_LINE_Y2 = 0, TARGET_HEIGHT

    CONF_THRESHOLD = 0.05
    MIN_BOX_AREA = 50
    MAX_BOX_AREA = 15000
    MIN_DISPLACEMENT = 3
    TIME_WINDOW = 0.5
    BOX_PERSISTENCE_TIME = 2.0  # Seconds to persist bounding boxes after last detection

    THUMB_SIZE = 50
    MAX_THUMBS = 10

    # ----------------------- Initialization ----------------------- #

    def init_device():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        return device

    def load_model(path, device):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {os.path.abspath(path)}")
        return YOLO(path).to(device)

    def init_zmq_publisher(port):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 1000)
        socket.setsockopt(zmq.SNDHWM, 10)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        for attempt in range(5):
            try:
                socket.bind(f"tcp://127.0.0.1:{port}")
                print(f"[ZMQ] Bound to tcp://127.0.0.1:{port}")
                return context, socket
            except zmq.error.ZMQError as e:
                print(f"[ZMQ] Bind failed (attempt {attempt + 1}): {e}")
                time.sleep(1)
        raise RuntimeError("ZMQ: Failed to bind socket.")

    def open_rtsp_stream(url):
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise ValueError("[ERROR] Could not open RTSP stream.")
        return cap

    # ----------------------- Utilities ----------------------- #

    def resize_and_center(frame, target_w, target_h):
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        return cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT)

    def draw_counting_line(frame, x, y1, y2, color=(39, 51, 63), thickness=2):
        cv2.line(frame, (x, y1), (x, y2), color, thickness)

    def draw_text(frame, text, x, y, color, scale=0.7, thickness=2):
        cv2.putText(frame, text.upper(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def create_bottom_strip(width, thumbs_left, thumbs_right, thumb_size, max_thumbs):
        strip = np.zeros((thumb_size, width, 3), dtype=np.uint8)
        for i, thumb in enumerate(thumbs_left[:max_thumbs]):
            x = i * thumb_size
            strip[0:thumb_size, x:x + thumb_size] = thumb
        for i, thumb in enumerate(thumbs_right[:max_thumbs]):
            x = width - (i + 1) * thumb_size
            strip[0:thumb_size, x:x + thumb_size] = thumb
        return strip

    # ----------------------- ZMQ Sender Thread ----------------------- #

    def zmq_send_thread(socket, queue):
        while True:
            try:
                msg_type, data = queue.get(timeout=1)
                if msg_type == "count":
                    socket.send_multipart([b"count", *data])
                elif msg_type == "image":
                    socket.send_multipart([b"image", data])
            except Empty:
                continue
            except Exception as e:
                print(f"[ZMQ Send Error] {e}")

    # ----------------------- Main ----------------------- #

    def main():
        device = init_device()
        model = load_model(MODEL_PATH, device)
        context, socket = init_zmq_publisher(ZMQ_PORT)
        send_queue = Queue()
        threading.Thread(target=zmq_send_thread, args=(socket, send_queue), daemon=True).start()

        cap = open_rtsp_stream(RTSP_URL)

        track_last_positions = defaultdict(lambda: None)
        track_timestamps = defaultdict(float)
        track_boxes = defaultdict(dict)  # Store bounding box info with timestamps

        prev_time = time.time()

        try:
            while True:
                current_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("[RTSP] Frame read failed. Reconnecting...")
                    cap.release()
                    time.sleep(2)
                    cap = open_rtsp_stream(RTSP_URL)
                    continue

                # FPS
                frame_time = current_time - prev_time
                prev_time = current_time
                fps = 1.0 / frame_time if frame_time > 0 else 0

                frame = resize_and_center(frame, TARGET_WIDTH, TARGET_HEIGHT)
                left_thumbs = []
                right_thumbs = []

                # Reset counts for this frame and track unique IDs
                inside_count = 0
                outside_count = 0
                seen_ids = set()

                results = model.track(source=frame, persist=True, conf=CONF_THRESHOLD, iou=0.5, tracker=TRACKER_CONFIG)
                boxes = results[0].boxes
                names = results[0].names

                # Update bounding box history with current detections
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        track_id = int(box.id[0]) if box.id is not None else None

                        label = names.get(cls_id, "").lower()
                        if track_id is None or "chicken" not in label:
                            continue

                        box_area = (x2 - x1) * (y2 - y1)
                        if not (MIN_BOX_AREA < box_area < MAX_BOX_AREA):
                            continue

                        # Store bounding box info with timestamp
                        track_boxes[track_id] = {
                            'xyxy': (x1, y1, x2, y2),
                            'timestamp': current_time,
                            'label': label
                        }

                # Process and draw persistent bounding boxes
                for track_id in list(track_boxes.keys()):
                    box_info = track_boxes[track_id]
                    if current_time - box_info['timestamp'] > BOX_PERSISTENCE_TIME:
                        del track_boxes[track_id]  # Remove expired boxes
                        continue

                    x1, y1, x2, y2 = box_info['xyxy']
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)

                    # Skip if this ID has already been counted in this frame
                    if track_id in seen_ids:
                        continue
                    seen_ids.add(track_id)

                    last_pos = track_last_positions[track_id]
                    last_time = track_timestamps[track_id]
                    is_moving = False
                    if last_pos and (current_time - last_time) <= TIME_WINDOW:
                        dx, dy = center[0] - last_pos[0], center[1] - last_pos[1]
                        if np.hypot(dx, dy) >= MIN_DISPLACEMENT:
                            is_moving = True
                    track_last_positions[track_id] = center
                    track_timestamps[track_id] = current_time
                    if not is_moving:
                        continue

                    # Count based on current position relative to the line
                    if center[0] > COUNTING_LINE_X:
                        inside_count += 1
                        print(f"[COUNT] CHICKEN on right (inside) | IN: {inside_count}, OUT: {outside_count}")
                    else:
                        outside_count += 1
                        print(f"[COUNT] CHICKEN on left (outside) | IN: {inside_count}, OUT: {outside_count}")

                    # Draw persistent bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    draw_text(frame, "CHICKEN", x1, y1 - 5, (0, 255, 0))

                    crop = frame[max(y1, 0):min(y2, frame.shape[0]), max(x1, 0):min(x2, frame.shape[1])]
                    if crop.size > 0:
                        thumb = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE))
                        if center[0] > COUNTING_LINE_X:
                        right_thumbs.append(thumb)
                        else:
                            left_thumbs.append(thumb)

                # Visuals
                draw_counting_line(frame, COUNTING_LINE_X, COUNTING_LINE_Y1, COUNTING_LINE_Y2)
                draw_text(frame, f"FPS: {fps:.2f}", 10, 20, (255, 255, 0))
                draw_text(frame, f"OUTSIDE: {outside_count}", 10, 330, (0, 0, 255))
                text_size, _ = cv2.getTextSize(f"INSIDE: {inside_count}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                draw_text(frame, f"INSIDE: {inside_count}", TARGET_WIDTH - text_size[0] - 10, 330, (0, 255, 0))

                bottom_strip = create_bottom_strip(TARGET_WIDTH, left_thumbs, right_thumbs, THUMB_SIZE, MAX_THUMBS)
                combined_frame = np.vstack((frame, bottom_strip))

                # ZMQ Send
                send_queue.put(("count", [str(inside_count).encode(), str(outside_count).encode()]))
                _, buffer = cv2.imencode(".jpg", combined_frame)
                send_queue.put(("image", buffer.tobytes()))

                time.sleep(0.03)

        finally:
            cap.release()
            socket.close()
            context.term()
            print("[INFO] Stream and socket closed.")

    if __name__ == "__main__":
        main()


   
