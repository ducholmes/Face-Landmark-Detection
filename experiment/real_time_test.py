import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
import time
from ultralytics import YOLO

print("Đang tải YOLOv11 Face Detector...")
try:
    model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    yolo_model = YOLO(model_path)
except Exception as e:
    print(f"Lỗi tải YOLO: {e}")
    print("Mẹo: Đảm bảo bạn đã tải file 'yolov11n-face.pt' về cùng thư mục.")
    exit()

HRNET_ONNX_PATH = "./experiment/hrnet_w18.onnx"
HRNET_INPUT_SIZE = (256, 256)
NUM_LANDMARKS = 98 

print(f"Đang tải HRNet ONNX từ: {HRNET_ONNX_PATH}...")
session = ort.InferenceSession(HRNET_ONNX_PATH, providers=['CPUExecutionProvider'])
hrnet_input_name = session.get_inputs()[0].name
print("Cả hai mô hình đã sẵn sàng.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

start_time = time.time()
frame_count = 0
fps = 0

print("Bắt đầu realtime. Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(frame, conf=0.5, verbose=False)
    
    output_frame = frame.copy()

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        x1, y1, x2, y2 = xyxy
        w = x2 - x1
        h = y2 - y1

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(output_frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        pad_w = int(w * 0.1)
        pad_h = int(h * 0.1)
        
        crop_y1 = max(0, y1 - pad_h)
        crop_y2 = min(frame.shape[0], y2 + pad_h)
        crop_x1 = max(0, x1 - pad_w)
        crop_x2 = min(frame.shape[1], x2 + pad_w)

        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w, _ = face_crop.shape

        if crop_h > 0 and crop_w > 0:
            img_pre = cv2.resize(face_crop, HRNET_INPUT_SIZE)
            img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
            img_pre = img_pre.astype(np.float32) / 255.0
            img_pre = img_pre.transpose(2, 0, 1)
            img_pre = np.expand_dims(img_pre, axis=0)

            hrnet_outputs = session.run(None, {hrnet_input_name: img_pre})[0]

            HEATMAP_SIZE = (64, 64)
            heatmaps = hrnet_outputs[0]
            flat_heatmaps = heatmaps.reshape(NUM_LANDMARKS, -1)
            max_idx = np.argmax(flat_heatmaps, axis=1)
            preds_y = max_idx // HEATMAP_SIZE[1]
            preds_x = max_idx % HEATMAP_SIZE[1]

            scale_heatmap_to_crop_h = crop_h / HEATMAP_SIZE[0]
            scale_heatmap_to_crop_w = crop_w / HEATMAP_SIZE[1]
            
            landmarks_on_crop_x = preds_x * scale_heatmap_to_crop_w
            landmarks_on_crop_y = preds_y * scale_heatmap_to_crop_h

            landmarks_final_x = landmarks_on_crop_x + crop_x1
            landmarks_final_y = landmarks_on_crop_y + crop_y1

            for i in range(NUM_LANDMARKS):
                cx = int(landmarks_final_x[i])
                cy = int(landmarks_final_y[i])
                cv2.circle(output_frame, (cx, cy), 1, (0, 255, 0), -1)

    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = time.time()
        frame_count = 0
    
    cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("YOLOv11 -> HRNet Landmark Detection (CPU)", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Đã dừng.")