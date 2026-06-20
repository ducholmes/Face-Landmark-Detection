# Face Landmark Detection & Real-time Tracking (98 Keypoints)

Dự án này triển khai một hệ thống phát hiện và bám vết 98 điểm mốc trên khuôn mặt (facial landmarks) theo thời gian thực. Hệ thống sử dụng kiến trúc kết hợp giữa **YOLOv11** (để phát hiện khuôn mặt) và **ResNet-18** (để hồi quy tọa độ điểm mốc), cùng với các bộ lọc tiên tiến như **1 Euro Filter** và **Deadzone Filter** để đảm bảo chuyển động mượt mà, không bị rung giật (jittering) trên luồng video.

## 1. Dataset & Data Processing

### Dataset
Dự án sử dụng bộ dữ liệu **WFLW (Wider Facial Landmarks in-the-wild)**. Đây là bộ dữ liệu phong phú với 98 điểm mốc khuôn mặt, bao gồm nhiều biểu cảm, góc nghiêng, điều kiện ánh sáng và che khuất phức tạp.

### Xử lý dữ liệu (Data Processing)
Pipeline tiền xử lý dữ liệu được thiết kế tối ưu hóa cho bài toán hồi quy (regression):
- **Cắt khuôn mặt có đệm (Crop with padding):** Sử dụng bounding box để cắt vùng khuôn mặt và mở rộng lề (padding) nhằm giữ lại toàn bộ chi tiết khuôn mặt mà không bị mất mép khuôn mặt.
- **Chuẩn hóa (Normalization):** Cắt vùng ảnh và resize về kích thước 256x256. Tọa độ các điểm mốc (keypoints) được chuẩn hóa về khoảng tương đối `[0, 1]` so với kích thước ảnh cắt để dễ dàng dự đoán.
- **Augmentation (Tăng cường dữ liệu):** Sử dụng thư viện `Albumentations` để áp dụng các phép biến đổi ngẫu nhiên như ShiftScaleRotate, RandomBrightnessContrast... giúp mô hình hoạt động ổn định và chống overfitting.

## 2. Kiến trúc mô hình (Model Architecture)

Hệ thống tập trung vào luồng xử lý tinh gọn, hiệu quả cao thay vì các kiến trúc heatmap cồng kềnh:
- **Face Detection:** Tích hợp bộ phát hiện khuôn mặt **YOLOv11** để tìm vùng khuôn mặt nhanh chóng trên khung hình.
- **Landmark Regression:** Sử dụng mạng backbone **ResNet-18** (`FaceLandmarkResNet`) được tùy chỉnh để trực tiếp hồi quy ra một vector tọa độ (98 điểm x 2 = 196 giá trị liên tục).
- **Quy trình Huấn luyện:** Quản lý quy trình qua **PyTorch Lightning** và **Hydra**. Mô hình được tối ưu hóa bằng hàm lỗi **Wing Loss** (một hàm mất mát chuyên dụng cho landmark giúp xử lý tốt các sai số nhỏ) kết hợp cùng thuật toán tối ưu AdamW và Cosine Annealing learning rate schedule để giảm NME (Normalized Mean Error).

## 3. Hậu xử lý & Lọc nhiễu (Post-processing)

Để ứng dụng chạy mượt mà theo thời gian thực qua giao diện Streamlit, hai cơ chế lọc nhiễu đã được áp dụng chuỗi cho tọa độ dự đoán của mạng nơ-ron:

- **1 Euro Filter:** Bộ lọc tín hiệu thông thấp (low-pass filter) tự động thích ứng với tốc độ chuyển động. Nó có khả năng lọc sạch nhiễu tần số cao để làm mịn chuyển động khi khuôn mặt đứng yên, nhưng vẫn giữ được độ trễ cực thấp (low latency) khi người dùng di chuyển nhanh.
- **Dead Zone Filter (Bộ lọc vùng chết):** Kỹ thuật này giữ nguyên vị trí hiển thị của một landmark nếu độ dịch chuyển (khoảng cách Euclidean) giữa frame mới và frame trước đó nhỏ hơn một ngưỡng pixel nhất định (`DEAD_ZONE_THRESHOLD`). Việc áp dụng bộ lọc vùng chết sau One Euro Filter giúp triệt tiêu hoàn toàn hiện tượng rung giật vi mô (micro-jittering) và hiện tượng "nháy" điểm mốc trên màn hình, mang lại cảm giác cực kỳ tự nhiên.


## Cài đặt & Chạy ứng dụng

Trước tiên, hãy cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

Sau khi cài đặt xong, bạn có thể khởi chạy ứng dụng Streamlit bằng lệnh:
```bash
streamlit run app/onnx_detect.py
```
