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

RESNET_ONNX_PATH   = str(ROOT_PATH / "experiment" / "resnet18.onnx")
INPUT_SIZE         = (256, 256)
NUM_LANDMARKS      = 98
SNAPSHOT_DIR       = "snapshots"

PAD_RATIO          = 0.2
LM_FCMIN           = 0.001
LM_BETA            = 0.03
BOX_FCMIN          = 0.001
BOX_BETA           = 0.03
BOUNDARY_THRESHOLD = 0.05

# --- DEAD ZONE FILTER ---
# Giữ nguyên landmark nếu khoảng cách Euclidean so với vị trí đã commit < threshold (pixels)
# Tuning: 1.0-1.5 (camera tốt), 2.0 (cân bằng), 3.0-4.0 (model jitter nhiều)
DEAD_ZONE_THRESHOLD = 2.0

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

for key, default in {
    'take_snapshot':    False,
    'snapshot_message': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def draw_bug_eye_glasses(frame, landmarks):
    left_pts  = landmarks[60:68]
    right_pts = landmarks[68:76]
    lc = left_pts.mean(axis=0).astype(int)
    rc = right_pts.mean(axis=0).astype(int)

    eye_dist = int(np.hypot(*(lc - rc)))
    r        = max(int(eye_dist * 0.55), 10)
    angle    = np.arctan2(lc[1] - rc[1], lc[0] - rc[0])

    bridge_l = (int(lc[0] - r * np.cos(angle)), int(lc[1] - r * np.sin(angle)))
    bridge_r = (int(rc[0] + r * np.cos(angle)), int(rc[1] + r * np.sin(angle)))
    cv2.line(frame, bridge_l, bridge_r, (0, 180, 255), max(r // 5, 3))

    perp_angle  = angle + np.pi / 2
    antenna_len = int(r * 1.4)
    dot_r       = max(r // 5, 4)

    for center in (lc, rc):
        cx, cy = int(center[0]), int(center[1])

        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.circle(frame, (cx, cy), r, (0, 200, 255), max(r // 4, 4))
        cv2.circle(frame, (cx, cy), r - max(r // 4, 4) // 2, (0, 0, 0), max(r // 8, 2))
        cv2.circle(frame, (int(cx - r * 0.3), int(cy - r * 0.3)), max(r // 5, 3), (255, 255, 255), -1)

        base_x = int(cx - r * 0.2 * np.cos(angle))
        base_y = int(cy - r * 0.2 * np.sin(angle) - r)
        tip_x  = int(base_x + antenna_len * np.cos(perp_angle - 0.3))
        tip_y  = int(base_y - antenna_len * abs(np.sin(perp_angle - 0.3)))
        cv2.line(frame, (base_x, base_y), (tip_x, tip_y), (0, 200, 255), max(r // 8, 2))
        cv2.circle(frame, (tip_x, tip_y), dot_r, (0, 80, 255), -1)
        cv2.circle(frame, (tip_x, tip_y), dot_r, (255, 255, 255), 1)

    return frame


def draw_clown_nose(frame, landmarks):
    nose_tip  = landmarks[54].astype(int)
    left_pts  = landmarks[60:68]
    right_pts = landmarks[68:76]
    eye_dist  = int(np.hypot(*(left_pts.mean(0) - right_pts.mean(0))))
    r         = max(int(eye_dist * 0.22), 8)
    cx, cy    = int(nose_tip[0]), int(nose_tip[1])

    shadow_overlay = frame.copy()
    cv2.ellipse(shadow_overlay, (cx + r // 6, cy + r // 4),
                (r, int(r * 0.6)), 0, 0, 360, (0, 0, 60), -1)
    cv2.addWeighted(shadow_overlay, 0.3, frame, 0.7, 0, frame)

    for i in range(r, 0, -1):
        ratio = 1.0 - i / r
        cv2.circle(frame, (cx, cy), i, (0, 0, int(80 + 175 * ratio)), 1)

    cv2.circle(frame, (cx, cy), r, (0, 0, 180), 2)
    cv2.ellipse(frame, (int(cx - r * 0.35), int(cy - r * 0.35)),
                (max(r // 3, 3), max(r // 5, 2)), -30, 0, 360, (255, 255, 255), -1)
    cv2.circle(frame, (int(cx - r * 0.15), int(cy - r * 0.15)), max(r // 7, 2), (255, 255, 255), -1)

    return frame


@st.cache_resource
def load_models():
    model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    yolo_model = YOLO(model_path)
    session    = ort.InferenceSession(RESNET_ONNX_PATH, providers=['CPUExecutionProvider'])
    return yolo_model, session


yolo_model, session = load_models()
input_name          = session.get_inputs()[0].name


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


def make_lm_filters(n):
    """Tạo OneEuro filters cho x, y và mảng lưu vị trí đã commit cho dead zone."""
    cfg = dict(freq=30, mincutoff=LM_FCMIN, beta=LM_BETA, dcutoff=1.0)
    return (
        [OneEuroFilter(**cfg) for _ in range(n)],   # filters_x
        [OneEuroFilter(**cfg) for _ in range(n)],   # filters_y
        np.full((n, 2), np.nan, dtype=np.float64),  # lm_committed: vị trí landmark đã commit
    )


def make_box_filters():
    cfg = dict(freq=30, mincutoff=BOX_FCMIN, beta=BOX_BETA, dcutoff=1.0)
    return [OneEuroFilter(**cfg) for _ in range(4)]


def apply_dead_zone(sx, sy, idx, lm_committed):
    """
    Áp dụng dead zone filter cho một landmark.
    Chỉ cập nhật vị trí committed nếu khoảng cách Euclidean >= DEAD_ZONE_THRESHOLD.

    Args:
        sx, sy: Toạ độ đã qua OneEuro filter (pixels)
        idx:    Chỉ số landmark
        lm_committed: Mảng (N, 2) lưu vị trí đã commit

    Returns:
        (out_x, out_y): Toạ độ output sau dead zone
    """
    prev = lm_committed[idx]

    if np.isnan(prev[0]):
        # Lần đầu tiên thấy landmark này: commit ngay
        lm_committed[idx] = [sx, sy]
    else:
        delta = np.hypot(sx - prev[0], sy - prev[1])
        if delta >= DEAD_ZONE_THRESHOLD:
            lm_committed[idx] = [sx, sy]
        # Nếu delta < threshold: giữ nguyên prev (không cập nhật)

    return lm_committed[idx][0], lm_committed[idx][1]


def save_snapshot(frame_bgr):
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filepath = os.path.join(SNAPSHOT_DIR, f"snapshot_{time_str}.jpg")
    if not cv2.imwrite(filepath, frame_bgr):
        raise RuntimeError(f"cv2.imwrite thất bại: {filepath}")
    return filepath


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
    dead_zone_px = st.slider(
        "Dead Zone (pixels)",
        min_value=0.0,
        max_value=6.0,
        value=DEAD_ZONE_THRESHOLD,
        step=0.5,
        help="Landmark giữ nguyên nếu chuyển động < N pixels. Tăng để ổn định hơn khi đứng yên."
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
        [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')], reverse=True)
    if snapshots:
        st.markdown("---")
        st.markdown(f"**📁 Snapshots ({len(snapshots)} ảnh)**")
        for fname in snapshots[:5]:
            img = cv2.imread(os.path.join(SNAPSHOT_DIR, fname))
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption=fname, use_container_width=True)

frame_window = st.image([])

if run_camera:
    cap = cv2.VideoCapture(0)

    filters_x, filters_y, lm_committed = make_lm_filters(NUM_LANDMARKS)
    box_filters                         = make_box_filters()

    start_time  = time.time()
    frame_count = 0
    fps         = 0

    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Lỗi: Không thể đọc từ camera.")
            break

        frame            = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        output_frame     = frame.copy()
        current_time     = time.time()

        # --- 1. YOLO DETECT MỖI FRAME ---
        results    = yolo_model.predict(frame, conf=0.5, verbose=False)
        target_box = None
        if len(results[0].boxes) > 0:
            target_box = results[0].boxes[0].xyxy[0].cpu().numpy()

        # --- 2. SMOOTH BOX + INFERENCE ---
        if target_box is not None:
            s_box = [box_filters[i](float(target_box[i]), current_time) for i in range(4)]
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

                if not (np.any(preds_rel < BOUNDARY_THRESHOLD) or
                        np.any(preds_rel > (1 - BOUNDARY_THRESHOLD))):

                    lx, ly = decode_coordinates(outputs, cx1, cy1, cx2, cy2)

                    # --- 3. ONEURO + DEAD ZONE FILTER ---
                    smoothed_pts = []
                    for i in range(NUM_LANDMARKS):
                        # Bước 1: OneEuro làm mượt chuyển động nhanh/chậm
                        sx = filters_x[i](float(lx[i]), current_time)
                        sy = filters_y[i](float(ly[i]), current_time)

                        # Bước 2: Dead zone — giữ nguyên nếu chuyển động < dead_zone_px
                        out_x, out_y = apply_dead_zone(sx, sy, i, lm_committed)
                        smoothed_pts.append([out_x, out_y])

                    landmarks = np.array(smoothed_pts)

                    # --- 4. VẼ FILTER ---
                    if filter_mode == "Chỉ hiện 98 Landmarks":
                        for sx, sy in landmarks:
                            cv2.circle(output_frame, (int(sx), int(sy)), 2, (0, 255, 0), -1)
                    elif filter_mode == "Kính Mắt Bọ 🐛":
                        output_frame = draw_bug_eye_glasses(output_frame, landmarks)
                    elif filter_mode == "Mũi Hề 🤡":
                        output_frame = draw_clown_nose(output_frame, landmarks)

                    cv2.rectangle(output_frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)
        else:
            # Reset filters khi mất face để tránh stale state
            box_filters                         = make_box_filters()
            filters_x, filters_y, lm_committed = make_lm_filters(NUM_LANDMARKS)

        # --- FPS ---
        frame_count += 1
        if frame_count >= 15:
            fps         = frame_count / (time.time() - start_time)
            start_time  = time.time()
            frame_count = 0
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), 2, 0.7, (0, 255, 255), 2)

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