import rootutils
import os
import datetime

# Setup root path để lấy đường dẫn tuyệt đối
ROOT_PATH = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
import time
from ultralytics import YOLO
import streamlit as st

# --- HẰNG SỐ CẤU HÌNH ---
RESNET_ONNX_PATH = str(ROOT_PATH / "experiment" / "resnet18.onnx")
INPUT_SIZE = (256, 256)
NUM_LANDMARKS = 98

PAD_RATIO = 0.3
LM_FCMIN = 0.001
LM_BETA = 0.007
BOX_FCMIN = 0.01
BOX_BETA = 0.007
DETECTION_INTERVAL = 60
BOUNDARY_THRESHOLD = 0.05

# --- KHỞI TẠO SESSION STATE ĐỂ LƯU ẢNH ---
if 'last_frame' not in st.session_state:
    st.session_state['last_frame'] = None

# --- CẤU HÌNH GIAO DIỆN TRANG WEB ---
st.set_page_config(page_title="98-Landmark Face App", page_icon="🎭", layout="wide")
st.title("🎭 98-Landmark Face Tracking & Filter App")

# --- HÀM LOAD MODEL ---
@st.cache_resource
def load_models():
    model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    yolo_model = YOLO(model_path)
    session = ort.InferenceSession(RESNET_ONNX_PATH, providers=['CPUExecutionProvider'])
    return yolo_model, session

yolo_model, session = load_models()
input_name = session.get_inputs()[0].name

# --- CÁC HÀM PHỤ TRỢ ---
def expand_bbox(x1, y1, x2, y2, frame_h, frame_w):
    w, h = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - w * PAD_RATIO))
    cx2 = min(frame_w, int(x2 + w * PAD_RATIO))
    cy1 = max(0, int(y1 - h * PAD_RATIO))
    cy2 = min(frame_h, int(y2 + h * PAD_RATIO))
    return cx1, cy1, cx2, cy2

def decode_coordinates(preds, cx1, cy1, cx2, cy2):
    preds = preds.reshape(-1, 2)
    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    lx = preds[:, 0] * crop_w + cx1
    ly = preds[:, 1] * crop_h + cy1
    return lx, ly

# --- ĐẢM BẢO THƯ MỤC LƯU ẢNH TỒN TẠI ---
os.makedirs("my_photos", exist_ok=True)

# --- MENU ĐIỀU KHIỂN BÊN TRÁI ---
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    run_camera = st.checkbox("Bật Camera", value=False)
    
    st.markdown("---")
    filter_mode = st.radio(
        "Chọn chế độ hiển thị:",
        ("Chỉ hiện 98 Landmarks", "Vẽ Kính (Mô phỏng)", "Vẽ Mũi Đỏ (Mô phỏng)")
    )
    
    st.markdown("---")
    take_pic = st.button("📸 Chụp Ảnh Ngay")
    
    # LOGIC CHỤP ẢNH MỚI
    if take_pic:
        if st.session_state['last_frame'] is not None:
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"my_photos/photo_{time_str}.jpg"
            cv2.imwrite(filename, st.session_state['last_frame'])
            st.success(f"📸 Đã lưu ảnh hoàn hảo vào: {filename}")
        else:
            st.warning("Chưa có khung hình nào để chụp! Hãy bật camera trước.")

# --- KHU VỰC HIỂN THỊ CAMERA ---
frame_window = st.image([])

if run_camera:
    cap = cv2.VideoCapture(0)
    
    filters_x = [OneEuroFilter(time.time(), 0, min_cutoff=LM_FCMIN, beta=LM_BETA) for _ in range(NUM_LANDMARKS)]
    filters_y = [OneEuroFilter(time.time(), 0, min_cutoff=LM_FCMIN, beta=LM_BETA) for _ in range(NUM_LANDMARKS)]
    box_filters = [OneEuroFilter(time.time(), 0, min_cutoff=BOX_FCMIN, beta=BOX_BETA) for _ in range(4)]

    is_tracking = False
    face_was_detected = False
    last_landmarks = None
    global_frame_counter = 0
    start_time = time.time()
    frame_count = 0
    fps = 0

    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret: 
            st.error("Lỗi: Không thể đọc từ camera.")
            break

        frame = cv2.flip(frame, 1)
        global_frame_counter += 1

        frame_h, frame_w = frame.shape[:2]
        output_frame = frame.copy()
        current_time = time.time()

        cx1, cy1, cx2, cy2 = 0, 0, 0, 0
        target_box = None

        should_detect = (not is_tracking) or (global_frame_counter % DETECTION_INTERVAL == 0)

        # --- 1. YOLO DETECTION ---
        if should_detect:
            results = yolo_model.predict(frame, conf=0.5, verbose=False)
            if len(results[0].boxes) > 0:
                target_box = results[0].boxes[0].xyxy[0].cpu().numpy()
                is_tracking = True
            else:
                is_tracking = False
        
        if not should_detect and is_tracking and last_landmarks is not None:
            x_min, y_min = np.min(last_landmarks, axis=0)
            x_max, y_max = np.max(last_landmarks, axis=0)
            target_box = [x_min, y_min, x_max, y_max]

        # --- 2. RESNET INFERENCE & ONE-EURO FILTER ---
        if target_box is not None:
            s_box = [box_filters[i](current_time, target_box[i]) for i in range(4)]
            cx1, cy1, cx2, cy2 = expand_bbox(s_box[0], s_box[1], s_box[2], s_box[3], frame_h, frame_w)

            face_crop = frame[cy1:cy2, cx1:cx2]
            if face_crop.size > 0:
                img_pre = cv2.resize(face_crop, INPUT_SIZE)
                img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
                img_pre = img_pre.astype(np.float32) / 255.0
                img_pre = (img_pre - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                img_pre = img_pre.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

                outputs = session.run(None, {input_name: img_pre})[0]
                preds_rel = outputs.reshape(-1, 2)

                out_of_bounds = np.any(preds_rel < BOUNDARY_THRESHOLD) or np.any(preds_rel > (1 - BOUNDARY_THRESHOLD))
                if out_of_bounds:
                    is_tracking = False

                lx, ly = decode_coordinates(outputs, cx1, cy1, cx2, cy2)

                if not face_was_detected:
                    for i in range(NUM_LANDMARKS):
                        filters_x[i].x_prev, filters_y[i].x_prev = lx[i], ly[i]
                    face_was_detected = True

                smoothed_pts = []
                for i in range(NUM_LANDMARKS):
                    sx = filters_x[i](current_time, lx[i])
                    sy = filters_y[i](current_time, ly[i])
                    smoothed_pts.append([sx, sy])
                    
                    if filter_mode == "Chỉ hiện 98 Landmarks":
                        cv2.circle(output_frame, (int(sx), int(sy)), 2, (0, 255, 0), -1)

                last_landmarks = np.array(smoothed_pts)

                # --- 3. VẼ MÔ PHỎNG FILTER ---
                if filter_mode == "Vẽ Kính (Mô phỏng)":
                    left_eye = (int(last_landmarks[60][0]), int(last_landmarks[60][1]))
                    right_eye = (int(last_landmarks[72][0]), int(last_landmarks[72][1]))
                    cv2.circle(output_frame, left_eye, 15, (255, 0, 0), -1) 
                    cv2.circle(output_frame, right_eye, 15, (255, 0, 0), -1)
                    cv2.line(output_frame, left_eye, right_eye, (255, 0, 0), 5) 

                elif filter_mode == "Vẽ Mũi Đỏ (Mô phỏng)":
                    nose = (int(last_landmarks[54][0]), int(last_landmarks[54][1]))
                    cv2.circle(output_frame, nose, 20, (0, 0, 255), -1)

                cv2.rectangle(output_frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)
        else:
            is_tracking = False
            face_was_detected = False

        # --- TÍNH FPS ---
        frame_count += 1
        if frame_count >= 15:
            fps = frame_count / (time.time() - start_time)
            start_time, frame_count = time.time(), 0

        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), 2, 0.7, (0, 255, 255), 2)

        # CẬP NHẬT FRAME VÀO SESSION STATE LIÊN TỤC
        st.session_state['last_frame'] = output_frame.copy()

        # --- HIỂN THỊ LÊN STREAMLIT ---
        frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

    cap.release()
else:
    st.info("👈 Hãy bật 'Bật Camera' ở bảng điều khiển bên trái để bắt đầu.")