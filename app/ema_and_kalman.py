import rootutils
import os
import datetime
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import time
from ultralytics import YOLO
import streamlit as st
from filterpy.kalman import KalmanFilter

ROOT_PATH = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

RESNET_ONNX_PATH   = str(ROOT_PATH / "experiment" / "resnet18.onnx")
INPUT_SIZE         = (256, 256)
NUM_LANDMARKS      = 98
SNAPSHOT_DIR       = "snapshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

if 'take_snapshot' not in st.session_state:
    st.session_state['take_snapshot'] = False
if 'target_id' not in st.session_state:
    st.session_state['target_id'] = None

# ===========================================================
# EMA FILTER (bbox)
# ===========================================================
class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

def make_box_filters():
    return [EMAFilter(0.3) for _ in range(4)]

# ===========================================================
# GLOBAL KALMAN FILTER
# ===========================================================
class FaceLandmarkKalmanFilter:
    def __init__(self, initial_landmarks, dt=1/30.0):
        self.num_points = initial_landmarks.shape[0]
        dim_x = 4 * self.num_points
        dim_z = 2 * self.num_points
        
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        self.kf.x = np.zeros((dim_x, 1))
        self.kf.x[0:2*self.num_points:2, 0] = initial_landmarks[:, 0]
        self.kf.x[1:2*self.num_points:2, 0] = initial_landmarks[:, 1]
        
        self.set_dt(dt)
        
        self.kf.H = np.zeros((dim_z, dim_x))
        for i in range(dim_z):
            self.kf.H[i, i] = 1.0
            
        self.kf.R = np.eye(dim_z) * 10.0
        self.kf.Q = np.eye(dim_x) * 0.1
        
        self.kf.P = np.eye(dim_x) * 10.0
        for i in range(2*self.num_points, 4*self.num_points):
            self.kf.P[i, i] = 100.0

    def set_dt(self, dt):
        dim_x = 4 * self.num_points
        self.kf.F = np.eye(dim_x)
        for i in range(2 * self.num_points):
            self.kf.F[i, i + 2*self.num_points] = dt

    def update(self, current_landmarks, dt):
        self.set_dt(dt)

        z = np.zeros((2 * self.num_points, 1))
        z[0::2, 0] = current_landmarks[:, 0]
        z[1::2, 0] = current_landmarks[:, 1]
        
        self.kf.predict()
        self.kf.update(z)
        
        smoothed = np.zeros((self.num_points, 2))
        smoothed[:, 0] = self.kf.x[0:2*self.num_points:2, 0]
        smoothed[:, 1] = self.kf.x[1:2*self.num_points:2, 0]
        
        return smoothed

# ===========================================================
# SIFT TRACKER — bổ sung khi YOLO mất track
# ===========================================================
class SIFTTracker:
    """
    Dùng SIFT + BFMatcher để tìm lại bbox khi YOLO không detect được face.

    Luồng hoạt động:
      1. Khi YOLO có bbox tốt → gọi update_reference() để lưu template.
      2. Khi YOLO mất track    → gọi estimate_box() để dùng SIFT local-search
         tìm vị trí face mới dựa trên keypoint matching giữa template và frame hiện tại.

    Local search: chỉ tìm kiếm trong vùng search_pad (pixels) quanh bbox cũ,
    giúp tránh false match từ nền và tăng tốc độ.
    """

    # Tham số Lucas-Kanade optical flow (dùng kết hợp với SIFT)
    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(
        self,
        n_features: int = 300,
        match_ratio: float = 0.75,
        min_matches: int = 6,
        search_pad: int = 80,
    ):
        """
        Args:
            n_features:  số SIFT keypoints tối đa trích xuất từ face template.
            match_ratio: Lowe's ratio test threshold.
            min_matches: số match tối thiểu để chấp nhận kết quả.
            search_pad:  padding (px) mở rộng vùng search quanh bbox cũ.
        """
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.bf   = cv2.BFMatcher(cv2.NORM_L2)

        self.match_ratio = match_ratio
        self.min_matches = min_matches
        self.search_pad  = search_pad

        # Trạng thái template
        self._ref_kps   = None   # List[KeyPoint] trong tọa độ face crop
        self._ref_desc  = None   # (N, 128) float32
        self._ref_box   = None   # (x1, y1, x2, y2) trong tọa độ frame gốc
        self._ref_gray  = None   # ảnh xám của face crop (để LK)
        self._lk_pts    = None   # điểm LK trong tọa độ frame gốc

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def update_reference(self, gray_frame: np.ndarray, box: tuple):
        """
        Lưu template SIFT từ face crop được xác định bởi box.

        Args:
            gray_frame: ảnh grayscale toàn frame (H, W).
            box:        (x1, y1, x2, y2) tọa độ face trong frame gốc.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        face_crop = gray_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return

        kps, desc = self.sift.detectAndCompute(face_crop, None)
        if desc is None or len(kps) < self.min_matches:
            return

        self._ref_kps  = kps
        self._ref_desc = desc
        self._ref_box  = (x1, y1, x2, y2)
        self._ref_gray = face_crop.copy()

        # Chuyển keypoints về tọa độ frame gốc để dùng LK
        pts = np.array([[kp.pt[0] + x1, kp.pt[1] + y1]
                        for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
        self._lk_pts = pts

    def estimate_box(self, gray_frame: np.ndarray) -> tuple | None:
        """
        Ước lượng bbox face mới trong gray_frame khi YOLO không detect được.

        Chiến lược 2 tầng:
          1. Lucas-Kanade sparse optical flow (nhanh) để ước lượng dịch chuyển.
          2. SIFT local-search trong vùng mở rộng quanh bbox cũ (chính xác hơn).

        Returns:
            (x1, y1, x2, y2) ước lượng, hoặc None nếu không đủ tin cậy.
        """
        if self._ref_desc is None or self._ref_box is None:
            return None

        # --- Tầng 1: Lucas-Kanade optical flow ---
        lk_box = self._lk_estimate(gray_frame)

        # Dùng kết quả LK làm điểm khởi đầu cho local SIFT search
        search_box = lk_box if lk_box is not None else self._ref_box

        # --- Tầng 2: SIFT local search ---
        sift_box = self._sift_local_search(gray_frame, search_box)

        # Cập nhật LK points nếu SIFT thành công
        if sift_box is not None:
            self._ref_box = sift_box
            self._update_lk_pts_from_box(gray_frame, sift_box)
            return sift_box

        # Fallback: chỉ dùng LK
        if lk_box is not None:
            self._ref_box = lk_box
            return lk_box

        return None

    def reset(self):
        """Xoá toàn bộ trạng thái tracker."""
        self._ref_kps  = None
        self._ref_desc = None
        self._ref_box  = None
        self._ref_gray = None
        self._lk_pts   = None

    # ----------------------------------------------------------
    # PRIVATE HELPERS
    # ----------------------------------------------------------

    def _lk_estimate(self, gray_frame: np.ndarray) -> tuple | None:
        """
        Dùng Lucas-Kanade để ước tính dịch chuyển bbox.
        Trả về bbox mới hoặc None nếu thất bại.
        """
        if self._lk_pts is None or len(self._lk_pts) < 4:
            return None

        prev_gray = self._make_prev_gray(gray_frame)
        if prev_gray is None:
            return None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray_frame, self._lk_pts, None, **self.LK_PARAMS
        )
        if next_pts is None:
            return None

        good_mask = status.ravel() == 1
        if good_mask.sum() < 4:
            return None

        old_pts = self._lk_pts[good_mask].reshape(-1, 2)
        new_pts = next_pts[good_mask].reshape(-1, 2)

        # Ước lượng homography (affine) hoặc chỉ lấy median shift
        shift = np.median(new_pts - old_pts, axis=0)  # (dx, dy)
        dx, dy = shift

        x1, y1, x2, y2 = self._ref_box
        new_box = (
            int(np.clip(x1 + dx, 0, gray_frame.shape[1])),
            int(np.clip(y1 + dy, 0, gray_frame.shape[0])),
            int(np.clip(x2 + dx, 0, gray_frame.shape[1])),
            int(np.clip(y2 + dy, 0, gray_frame.shape[0])),
        )

        # Cập nhật LK points
        self._lk_pts = next_pts[good_mask].reshape(-1, 1, 2)

        return new_box

    def _sift_local_search(self, gray_frame: np.ndarray, search_box: tuple) -> tuple | None:
        """
        Trích SIFT trong vùng search (search_box + padding), match với template,
        rồi dùng homography để tính bbox mới.

        Local search giới hạn vùng tìm kiếm → nhanh hơn và ít false positive.
        """
        if self._ref_desc is None:
            return None

        h_frame, w_frame = gray_frame.shape[:2]
        x1, y1, x2, y2 = search_box
        pad = self.search_pad

        # Vùng search mở rộng trong frame
        sx1 = max(0, x1 - pad)
        sy1 = max(0, y1 - pad)
        sx2 = min(w_frame, x2 + pad)
        sy2 = min(h_frame, y2 + pad)

        search_crop = gray_frame[sy1:sy2, sx1:sx2]
        if search_crop.size == 0:
            return None

        kps2, desc2 = self.sift.detectAndCompute(search_crop, None)
        if desc2 is None or len(kps2) < self.min_matches:
            return None

        # kNN match + Lowe's ratio test
        try:
            raw_matches = self.bf.knnMatch(self._ref_desc, desc2, k=2)
        except cv2.error:
            return None

        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.match_ratio * n.distance:
                    good.append(m)

        if len(good) < self.min_matches:
            return None

        # Điểm tương ứng: template (tọa độ crop) ↔ search region (tọa độ crop)
        src_pts = np.float32(
            [self._ref_kps[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kps2[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        # Tìm homography để transform bbox template → frame hiện tại
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        inliers = mask.ravel().sum()
        if inliers < self.min_matches:
            return None

        # Tính bbox template trong tọa độ crop template
        ref_x1, ref_y1, ref_x2, ref_y2 = self._ref_box
        tw = ref_x2 - ref_x1
        th = ref_y2 - ref_y1
        template_corners = np.float32(
            [[0, 0], [tw, 0], [tw, th], [0, th]]
        ).reshape(-1, 1, 2)

        # Biến đổi qua homography → tọa độ trong search crop
        dst_corners = cv2.perspectiveTransform(template_corners, H)
        if dst_corners is None:
            return None

        dst_corners = dst_corners.reshape(-1, 2)

        # Chuyển về tọa độ frame gốc (cộng offset của search region)
        dst_corners[:, 0] += sx1
        dst_corners[:, 1] += sy1

        nx1 = int(np.clip(dst_corners[:, 0].min(), 0, w_frame))
        ny1 = int(np.clip(dst_corners[:, 1].min(), 0, h_frame))
        nx2 = int(np.clip(dst_corners[:, 0].max(), 0, w_frame))
        ny2 = int(np.clip(dst_corners[:, 1].max(), 0, h_frame))

        # Kiểm tra bbox hợp lệ (không quá nhỏ hoặc bị đảo)
        if nx2 - nx1 < 20 or ny2 - ny1 < 20:
            return None

        return (nx1, ny1, nx2, ny2)

    def _make_prev_gray(self, gray_frame: np.ndarray) -> np.ndarray | None:
        """Tạo ảnh xám "frame trước" để chạy LK từ toàn frame."""
        # Dùng frame hiện tại khi chưa có prev — LK sẽ tự xử lý
        if self._ref_gray is None:
            return None
        h, w = gray_frame.shape[:2]
        x1, y1, x2, y2 = self._ref_box
        prev = np.zeros_like(gray_frame)
        rh, rw = self._ref_gray.shape[:2]
        # Paste ref crop vào đúng vị trí
        pe_y2 = min(y1 + rh, h)
        pe_x2 = min(x1 + rw, w)
        prev[y1:pe_y2, x1:pe_x2] = self._ref_gray[:pe_y2-y1, :pe_x2-x1]
        return prev

    def _update_lk_pts_from_box(self, gray_frame: np.ndarray, box: tuple):
        """Sau khi SIFT update bbox, dùng goodFeaturesToTrack để lấy LK points mới."""
        x1, y1, x2, y2 = box
        face_crop = gray_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return
        pts = cv2.goodFeaturesToTrack(face_crop, maxCorners=200,
                                       qualityLevel=0.01, minDistance=5)
        if pts is not None:
            pts[:, 0, 0] += x1
            pts[:, 0, 1] += y1
            self._lk_pts = pts

# ===========================================================
# HELPERS
# ===========================================================
def expand_bbox(x1, y1, x2, y2, frame_h, frame_w):
    w = x2 - x1
    h = y2 - y1
    size = max(w, h)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    half = size / 2

    x1_new = int(max(0, cx - half))
    y1_new = int(max(0, cy - half))
    x2_new = int(min(frame_w, cx + half))
    y2_new = int(min(frame_h, cy + half))

    return x1_new, y1_new, x2_new, y2_new

def decode_coordinates(preds, cx1, cy1, cx2, cy2):
    preds = preds.reshape(-1, 2)
    lx = preds[:, 0] * (cx2 - cx1) + cx1
    ly = preds[:, 1] * (cy2 - cy1) + cy1
    return lx, ly

def save_snapshot(frame):
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{time_str}.jpg")
    cv2.imwrite(path, frame)

# ===========================================================
# LOAD MODEL
# ===========================================================
@st.cache_resource
def load_models():
    model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    yolo_model = YOLO(model_path)
    session = ort.InferenceSession(RESNET_ONNX_PATH, providers=['CPUExecutionProvider'])
    return yolo_model, session

yolo_model, session = load_models()
input_name = session.get_inputs()[0].name

# ===========================================================
# UI
# ===========================================================
st.set_page_config(page_title="Face Detect App", page_icon="🎭", layout="wide")
st.title("🎭 98-Landmark Face Detection")

with st.sidebar:
    run_camera = st.checkbox("Bật Camera", value=False)
    filter_mode = st.radio("Chế độ:", ("Landmarks", "Bug Eye", "Clown Nose"))

    if st.button("📸 Chụp Ảnh"):
        st.session_state['take_snapshot'] = True

frame_window = st.image([])

# ===========================================================
# CAMERA LOOP
# ===========================================================
if run_camera:
    cap = cv2.VideoCapture(0)

    b_filters   = make_box_filters()
    face_kalman = None
    sift_tracker = SIFTTracker(
        n_features=300,
        match_ratio=0.75,
        min_matches=6,
        search_pad=80,
    )
    prev_time = time.time()

    # Đếm số frame liên tiếp YOLO không detect được
    yolo_miss_count = 0
    # Ngưỡng: sau bao nhiêu frame miss mới coi là mất track thật sự
    YOLO_MISS_THRESHOLD = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        h, w     = frame.shape[:2]
        output   = frame.copy()
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results  = yolo_model.track(frame, persist=True, conf=0.5, verbose=False)

        # --- Lấy bbox từ YOLO hoặc fallback sang SIFT ---
        target_box_raw  = None   # bbox thô từ YOLO (float)
        used_sift       = False  # để vẽ indicator

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes  = results[0].boxes
            ids    = boxes.id.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()

            if st.session_state['target_id'] is None:
                st.session_state['target_id'] = ids[0]

            for i, id_ in enumerate(ids):
                if id_ == st.session_state['target_id']:
                    target_box_raw = coords[i]
                    break

        if target_box_raw is not None:
            # YOLO detect thành công
            yolo_miss_count = 0

            # Cập nhật template SIFT từ bbox YOLO mới nhất
            raw_x1, raw_y1, raw_x2, raw_y2 = [int(v) for v in target_box_raw]
            sift_tracker.update_reference(gray, (raw_x1, raw_y1, raw_x2, raw_y2))

        else:
            # YOLO không tìm thấy → thử SIFT local search
            yolo_miss_count += 1

            if yolo_miss_count <= YOLO_MISS_THRESHOLD:
                # Trong ngưỡng chịu đựng → hỏi SIFT
                sift_box = sift_tracker.estimate_box(gray)
                if sift_box is not None:
                    target_box_raw = np.array(sift_box, dtype=np.float32)
                    used_sift = True

            if target_box_raw is None:
                # Vượt ngưỡng hoặc SIFT cũng thất bại → reset hoàn toàn
                if yolo_miss_count > YOLO_MISS_THRESHOLD:
                    face_kalman = None
                    b_filters   = make_box_filters()
                    sift_tracker.reset()
                    st.session_state['target_id'] = None
                    yolo_miss_count = 0

        # --- Xử lý bbox và landmark ---
        if target_box_raw is not None:
            # EMA smoothing trên bbox
            sb   = [b_filters[i].update(float(target_box_raw[i])) for i in range(4)]
            x1, y1, x2, y2 = expand_bbox(sb[0], sb[1], sb[2], sb[3], h, w)

            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                img = cv2.resize(face, INPUT_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = img.transpose(2, 0, 1)[np.newaxis]

                preds        = session.run(None, {input_name: img})[0]
                lx, ly       = decode_coordinates(preds, x1, y1, x2, y2)
                raw_lms      = np.stack([lx, ly], axis=1)

                # dt realtime
                curr_time = time.time()
                dt        = curr_time - prev_time
                prev_time = curr_time
                dt        = min(max(dt, 1e-3), 0.1)

                if face_kalman is None:
                    face_kalman = FaceLandmarkKalmanFilter(raw_lms)

                lms = face_kalman.update(raw_lms, dt)

                for p in lms:
                    cv2.circle(output, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)

                # Màu bbox: xanh dương = YOLO, vàng = SIFT fallback
                box_color = (0, 200, 255) if used_sift else (255, 0, 0)
                cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 1)

                # Hiển thị label nguồn tracking
                label = "SIFT" if used_sift else "YOLO"
                cv2.putText(output, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

        else:
            face_kalman = None

        if st.session_state.get('take_snapshot'):
            st.session_state['take_snapshot'] = False
            save_snapshot(output)

        frame_window.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    cap.release()