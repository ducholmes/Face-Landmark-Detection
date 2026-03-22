import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
import time
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

print("Đang tải YOLOv11 Face Detector...")
try:
    model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    yolo_model = YOLO(model_path)
except Exception as e:
    print(f"Lỗi tải YOLO: {e}")
    exit()

HRNET_ONNX_PATH = "./experiment/hrnet_w18.onnx"
HRNET_INPUT_SIZE = (256, 256)
NUM_LANDMARKS = 98
GAUSSIAN_SIGMA = 1.0

PAD_W_RATIO = 0.4
PAD_H_TOP_RATIO = 0.4
PAD_H_BOT_RATIO = 0.4

print(f"Đang tải HRNet ONNX từ: {HRNET_ONNX_PATH}...")
session = ort.InferenceSession(HRNET_ONNX_PATH, providers=['CPUExecutionProvider'])
hrnet_input_name = session.get_inputs()[0].name
print("Cả hai mô hình đã sẵn sàng.")


def expand_bbox(x1, y1, x2, y2, frame_h, frame_w):
    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * PAD_W_RATIO)
    pad_top = int(h * PAD_H_TOP_RATIO)
    pad_bot = int(h * PAD_H_BOT_RATIO)
    crop_x1 = max(0, x1 - pad_w)
    crop_x2 = min(frame_w, x2 + pad_w)
    crop_y1 = max(0, y1 - pad_top)
    crop_y2 = min(frame_h, y2 + pad_bot)
    return crop_x1, crop_y1, crop_x2, crop_y2


def decode_heatmaps_numpy(heatmaps, crop_x1, crop_y1, crop_x2, crop_y2,
                           sigma=GAUSSIAN_SIGMA):
    C, H, W = heatmaps.shape

    heatmaps = np.stack([gaussian_filter(heatmaps[c], sigma=sigma) for c in range(C)])

    flat = heatmaps.reshape(C, -1)
    max_idx = np.argmax(flat, axis=1)

    preds_x = (max_idx % W).astype(np.float32)
    preds_y = (max_idx // W).astype(np.float32)

    # Sub-pixel refinement
    for c in range(C):
        hm = heatmaps[c]
        px = int(preds_x[c])
        py = int(preds_y[c])
        if 0 < px < W - 1 and 0 < py < H - 1:
            diff_x = hm[py, px + 1] - hm[py, px - 1]
            diff_y = hm[py + 1, px] - hm[py - 1, px]
            preds_x[c] += np.sign(diff_x) * 0.25
            preds_y[c] += np.sign(diff_y) * 0.25

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    landmarks_x = (preds_x / W) * crop_w + crop_x1
    landmarks_y = (preds_y / H) * crop_h + crop_y1

    return landmarks_x, landmarks_y


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Không mở được webcam!")
    exit()

start_time = time.time()
frame_count = 0
fps = 0

print("Bắt đầu realtime. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame!")
        break

    frame_h, frame_w = frame.shape[:2]
    results = yolo_model.predict(frame, conf=0.5, verbose=False)
    output_frame = frame.copy()

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
            x1, y1, x2, y2, frame_h, frame_w
        )

        cv2.rectangle(output_frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 255), 2)

        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h_actual = crop_y2 - crop_y1
        crop_w_actual = crop_x2 - crop_x1

        if crop_h_actual > 0 and crop_w_actual > 0:
            img_pre = cv2.resize(face_crop, HRNET_INPUT_SIZE)
            img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
            img_pre = img_pre.astype(np.float32) / 255.0
            img_pre = img_pre.transpose(2, 0, 1)[np.newaxis]

            heatmaps = session.run(None, {hrnet_input_name: img_pre})[0][0]

            lm_x, lm_y = decode_heatmaps_numpy(
                heatmaps, crop_x1, crop_y1, crop_x2, crop_y2
            )

            for i in range(NUM_LANDMARKS):
                cx, cy = int(lm_x[i]), int(lm_y[i])
                cv2.circle(output_frame, (cx, cy), 1, (0, 255, 0), -1)

    frame_count += 1
    if frame_count >= 10:
        fps = frame_count / (time.time() - start_time)
        start_time = time.time()
        frame_count = 0

    cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLOv11 -> HRNet Landmark Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Đã dừng.")