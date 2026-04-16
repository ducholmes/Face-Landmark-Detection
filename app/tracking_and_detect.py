import rootutils
import os
import datetime

ROOT_PATH = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
import time
from ultralytics import YOLO
import streamlit as st
from OneEuroFilter import OneEuroFilter

# --- HẰNG SỐ CẤU HÌNH ---
RESNET_ONNX_PATH   = str(ROOT_PATH / "experiment" / "resnet18.onnx")
INPUT_SIZE         = (256, 256)
NUM_LANDMARKS      = 98
SNAPSHOT_DIR       = "snapshots"

PAD_RATIO          = 0.2
LM_FCMIN           = 0.001
LM_BETA            = 0.007
BOX_FCMIN          = 0.01
BOX_BETA           = 0.007
DETECTION_INTERVAL = 10
BOUNDARY_THRESHOLD = 0.05

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

for key, default in {
    'take_snapshot':    False,
    'snapshot_message': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ===========================================================
# VẼ FILTER BẰNG OPENCV (không cần file PNG)
# ===========================================================

def draw_bug_eye_glasses(frame: np.ndarray,
                         landmarks: np.ndarray) -> np.ndarray:
    """
    Kính mắt bọ: 2 tròng kính hình tròn to, tròng đen trong suốt,
    viền dày màu vàng/đen, có gọng nối và 2 râu ăng-ten.

    Vẽ trực tiếp lên frame bằng OpenCV — không cần file PNG.
    """
    # Tâm mỗi mắt
    left_pts   = landmarks[60:68]
    right_pts  = landmarks[68:76]
    lc = left_pts.mean(axis=0).astype(int)   # (x, y) mắt trái
    rc = right_pts.mean(axis=0).astype(int)  # (x, y) mắt phải

    # Khoảng cách 2 mắt → bán kính tròng kính
    eye_dist = int(np.hypot(*(lc - rc)))
    r        = max(int(eye_dist * 0.55), 10)  # bán kính mỗi tròng

    # Góc nghiêng đường nối 2 mắt (để xoay tâm kính nếu đầu nghiêng)
    angle = np.arctan2(lc[1] - rc[1], lc[0] - rc[0])

    # --- Gọng nối giữa 2 tròng ---
    bridge_l = (int(lc[0] - r * np.cos(angle)),
                int(lc[1] - r * np.sin(angle)))
    bridge_r = (int(rc[0] + r * np.cos(angle)),
                int(rc[1] + r * np.sin(angle)))
    cv2.line(frame, bridge_l, bridge_r, (0, 180, 255), max(r // 5, 3))

    # --- Vẽ 2 tròng kính ---
    for center in (lc, rc):
        cx, cy = int(center[0]), int(center[1])

        # Nền tròng: đen bán trong suốt (vẽ overlay)
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Viền ngoài vàng dày
        cv2.circle(frame, (cx, cy), r,      (0, 200, 255), max(r // 4, 4))
        # Viền trong đen mỏng
        cv2.circle(frame, (cx, cy), r - max(r // 4, 4) // 2,
                   (0, 0, 0), max(r // 8, 2))

        # Ánh sáng phản chiếu (highlight trắng nhỏ)
        hx = int(cx - r * 0.3)
        hy = int(cy - r * 0.3)
        cv2.circle(frame, (hx, hy), max(r // 5, 3), (255, 255, 255), -1)

    # --- Ăng-ten (râu trên đầu kính) ---
    perp_angle = angle + np.pi / 2          # vuông góc với đường nối
    antenna_len = int(r * 1.4)
    dot_r       = max(r // 5, 4)

    for center in (lc, rc):
        cx, cy = int(center[0]), int(center[1])
        # Gốc ăng-ten: đỉnh tròng
        base_x = int(cx - r * 0.2 * np.cos(angle))
        base_y = int(cy - r * 0.2 * np.sin(angle) - r)
        # Đầu ăng-ten: chệch sang một chút
        tip_x  = int(base_x + antenna_len * np.cos(perp_angle - 0.3))
        tip_y  = int(base_y - antenna_len * abs(np.sin(perp_angle - 0.3)))
        cv2.line(frame, (base_x, base_y), (tip_x, tip_y),
                 (0, 200, 255), max(r // 8, 2))
        cv2.circle(frame, (tip_x, tip_y), dot_r, (0, 80, 255), -1)
        cv2.circle(frame, (tip_x, tip_y), dot_r, (255, 255, 255), 1)

    return frame


def draw_clown_nose(frame: np.ndarray,
                    landmarks: np.ndarray) -> np.ndarray:
    """
    Mũi hề: hình cầu đỏ bóng với highlight trắng.
    Tâm = landmark 54 (đỉnh mũi).
    Kích thước tỉ lệ theo khoảng cách 2 mắt.
    """
    nose_tip  = landmarks[54].astype(int)
    left_pts  = landmarks[60:68]
    right_pts = landmarks[68:76]
    eye_dist  = int(np.hypot(*(left_pts.mean(0) - right_pts.mean(0))))
    r         = max(int(eye_dist * 0.22), 8)

    cx, cy = int(nose_tip[0]), int(nose_tip[1])

    # Bóng đổ (shadow) phía dưới
    shadow_overlay = frame.copy()
    cv2.ellipse(shadow_overlay, (cx + r // 6, cy + r // 4),
                (r, int(r * 0.6)), 0, 0, 360, (0, 0, 60), -1)
    cv2.addWeighted(shadow_overlay, 0.3, frame, 0.7, 0, frame)

    # Gradient đỏ: vẽ nhiều vòng tròn từ ngoài vào trong (tối→sáng)
    for i in range(r, 0, -1):
        ratio = 1.0 - i / r                      # 0 (ngoài) → 1 (trong)
        red   = int(80 + 175 * ratio)             # 80 → 255
        cv2.circle(frame, (cx, cy), i, (0, 0, red), 1)

    # Viền ngoài đỏ đậm
    cv2.circle(frame, (cx, cy), r, (0, 0, 180), 2)

    # Highlight trắng (phản sáng)
    hx = int(cx - r * 0.35)
    hy = int(cy - r * 0.35)
    cv2.ellipse(frame, (hx, hy),
                (max(r // 3, 3), max(r // 5, 2)), -30, 0, 360,
                (255, 255, 255), -1)
    # Highlight nhỏ hơn
    cv2.circle(frame,
               (int(cx - r * 0.15), int(cy - r * 0.15)),
               max(r // 7, 2), (255, 255, 255), -1)

    return frame


# --- LOAD MODEL ---
@st.cache_resource
def load_models():
    model_path = hf_hub_download(
        repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    yolo_model = YOLO(model_path)
    session    = ort.InferenceSession(
        RESNET_ONNX_PATH, providers=['CPUExecutionProvider'])
    return yolo_model, session


yolo_model, session = load_models()
input_name          = session.get_inputs()[0].name


# --- HÀM PHỤ TRỢ ---
def expand_bbox(x1, y1, x2, y2, frame_h, frame_w):
    w, h = x2 - x1, y2 - y1
    return (max(0,       int(x1 - w * PAD_RATIO)),
            max(0,       int(y1 - h * PAD_RATIO)),
            min(frame_w, int(x2 + w * PAD_RATIO)),
            min(frame_h, int(y2 + h * PAD_RATIO)))


def decode_coordinates(preds, cx1, cy1, cx2, cy2):
    preds = preds.reshape(-1, 2)
    lx    = preds[:, 0] * (cx2 - cx1) + cx1
    ly    = preds[:, 1] * (cy2 - cy1) + cy1
    return lx, ly


def make_lm_filters(n: int):
    cfg = dict(freq=30, mincutoff=LM_FCMIN, beta=LM_BETA, dcutoff=1.0)
    return ([OneEuroFilter(**cfg) for _ in range(n)],
            [OneEuroFilter(**cfg) for _ in range(n)])


def make_box_filters():
    cfg = dict(freq=30, mincutoff=BOX_FCMIN, beta=BOX_BETA, dcutoff=1.0)
    return [OneEuroFilter(**cfg) for _ in range(4)]


def save_snapshot(frame_bgr: np.ndarray) -> str:
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filepath = os.path.join(SNAPSHOT_DIR, f"snapshot_{time_str}.jpg")
    if not cv2.imwrite(filepath, frame_bgr):
        raise RuntimeError(f"cv2.imwrite thất bại: {filepath}")
    return filepath


# ===========================================================
# GIAO DIỆN
# ===========================================================
st.set_page_config(page_title="98-Landmark Face App", page_icon="🎭", layout="wide")
st.title("🎭 98-Landmark Face Tracking & Filter App")

with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    run_camera = st.checkbox("Bật Camera", value=False)

    st.markdown("---")
    filter_mode = st.radio(
        "Chọn chế độ hiển thị:",
        ("Chỉ hiện 98 Landmarks", "Kính Mắt Bọ 🐛", "Mũi Hề 🤡")
    )

    st.markdown("---")
    if st.button("📸 Chụp Ảnh", use_container_width=True, disabled=not run_camera):
        st.session_state['take_snapshot'] = True
    if not run_camera:
        st.caption("Bật camera để dùng chức năng chụp ảnh.")

    msg = st.session_state.get('snapshot_message')
    if msg:
        (st.success if msg[0] == 'success' else st.error)(msg[1])
        st.session_state['snapshot_message'] = None

    snapshots = sorted(
        [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')],
        reverse=True
    )
    if snapshots:
        st.markdown("---")
        st.markdown(f"**📁 Snapshots ({len(snapshots)} ảnh)**")
        for fname in snapshots[:5]:
            img = cv2.imread(os.path.join(SNAPSHOT_DIR, fname))
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption=fname, use_container_width=True)

frame_window = st.image([])

# ===========================================================
# VÒNG LẶP CAMERA
# ===========================================================
if run_camera:
    cap = cv2.VideoCapture(0)

    filters_x, filters_y = make_lm_filters(NUM_LANDMARKS)
    box_filters           = make_box_filters()

    is_tracking          = False
    face_was_detected    = False
    last_landmarks       = None
    global_frame_counter = 0
    start_time           = time.time()
    frame_count          = 0
    fps                  = 0

    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Lỗi: Không thể đọc từ camera.")
            break

        frame        = cv2.flip(frame, 1)
        global_frame_counter += 1
        frame_h, frame_w = frame.shape[:2]
        output_frame = frame.copy()
        current_time = time.time()

        target_box    = None
        should_detect = (not is_tracking) or (global_frame_counter % DETECTION_INTERVAL == 0)

        # --- 1. YOLO ---
        if should_detect:
            results = yolo_model.predict(frame, conf=0.5, verbose=False)
            if len(results[0].boxes) > 0:
                target_box  = results[0].boxes[0].xyxy[0].cpu().numpy()
                is_tracking = True
            else:
                is_tracking       = False
                face_was_detected = False
                last_landmarks    = None
                filters_x, filters_y = make_lm_filters(NUM_LANDMARKS)
                box_filters           = make_box_filters()

        if not should_detect and is_tracking and last_landmarks is not None:
            x_min, y_min = np.min(last_landmarks, axis=0)
            x_max, y_max = np.max(last_landmarks, axis=0)
            target_box   = [x_min, y_min, x_max, y_max]

        # --- 2. INFERENCE + ONE-EURO FILTER ---
        if target_box is not None:
            s_box = [box_filters[i](target_box[i], current_time) for i in range(4)]
            cx1, cy1, cx2, cy2 = expand_bbox(
                s_box[0], s_box[1], s_box[2], s_box[3], frame_h, frame_w)

            face_crop = frame[cy1:cy2, cx1:cx2]
            if face_crop.size > 0:
                img_pre = cv2.resize(face_crop, INPUT_SIZE)
                img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
                img_pre = img_pre.astype(np.float32) / 255.0
                img_pre = (img_pre - np.array([0.485, 0.456, 0.406])) \
                          / np.array([0.229, 0.224, 0.225])
                img_pre = img_pre.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

                outputs   = session.run(None, {input_name: img_pre})[0]
                preds_rel = outputs.reshape(-1, 2)

                if (np.any(preds_rel < BOUNDARY_THRESHOLD) or
                        np.any(preds_rel > (1 - BOUNDARY_THRESHOLD))):
                    is_tracking       = False
                    face_was_detected = False
                    last_landmarks    = None
                    filters_x, filters_y = make_lm_filters(NUM_LANDMARKS)
                    box_filters           = make_box_filters()
                else:
                    lx, ly = decode_coordinates(outputs, cx1, cy1, cx2, cy2)

                    if not face_was_detected:
                        for i in range(NUM_LANDMARKS):
                            filters_x[i](float(lx[i]), current_time)
                            filters_y[i](float(ly[i]), current_time)
                        face_was_detected = True

                    smoothed_pts = []
                    for i in range(NUM_LANDMARKS):
                        sx = filters_x[i](float(lx[i]), current_time)
                        sy = filters_y[i](float(ly[i]), current_time)
                        smoothed_pts.append([sx, sy])
                    last_landmarks = np.array(smoothed_pts)

                    # --- 3. ÁP DỤNG FILTER ---
                    if filter_mode == "Chỉ hiện 98 Landmarks":
                        for sx, sy in last_landmarks:
                            cv2.circle(output_frame,
                                       (int(sx), int(sy)), 2, (0, 255, 0), -1)

                    elif filter_mode == "Kính Mắt Bọ 🐛":
                        output_frame = draw_bug_eye_glasses(output_frame, last_landmarks)

                    elif filter_mode == "Mũi Hề 🤡":
                        output_frame = draw_clown_nose(output_frame, last_landmarks)

                    cv2.rectangle(output_frame,
                                  (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)
        else:
            is_tracking       = False
            face_was_detected = False

        # --- FPS ---
        frame_count += 1
        if frame_count >= 15:
            fps         = frame_count / (time.time() - start_time)
            start_time  = time.time()
            frame_count = 0
        cv2.putText(output_frame, f"FPS: {fps:.1f}",
                    (10, 30), 2, 0.7, (0, 255, 255), 2)

        # --- CHỤP ẢNH ---
        if st.session_state.get('take_snapshot'):
            st.session_state['take_snapshot'] = False
            try:
                saved = save_snapshot(output_frame)
                st.session_state['snapshot_message'] = ('success', f"✅ Đã lưu: {saved}")
            except Exception as e:
                st.session_state['snapshot_message'] = ('error', f"❌ Lỗi: {e}")

        frame_window.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.info("👈 Hãy bật 'Bật Camera' ở bảng điều khiển bên trái để bắt đầu.")