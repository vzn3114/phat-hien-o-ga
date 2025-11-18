# 🚗 Hệ Thống Phát Hiện Ổ Gà Theo Thời Gian Thực

Ứng dụng AI thực tế phát hiện ổ gà trên mặt đường sử dụng **YOLOv8** và cảnh báo âm thanh, giúp người lái xe phòng tránh rủi ro giao thông.

---

## 📋 Tính Năng Chính

✅ **Phát hiện real-time** - Xử lý video/webcam với tốc độ cao  
✅ **Giao diện Web đơn giản** - Dễ tiếp cận, không cần kiến thức IT  
✅ **Cảnh báo âm thanh** - Thông báo tức thì khi phát hiện ổ gà  
✅ **CLI script nâng cao** - Cho người dùng muốn tùy chỉnh chi tiết  
✅ **Hỗ trợ GPU** - Tốc độ xử lý nhanh với NVIDIA GPU  
✅ **Độ tin cậy có thể điều chỉnh** - Tối ưu theo điều kiện thực tế

---

## 🚀 Cài Đặt Nhanh (3 Bước)

### Bước 1: Cài đặt Python Dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Ứng Dụng

**Chọn một trong 3 cách:**

####  **Cách 1: Giao Diện Web**

```bash
python app.py
```

Mở trình duyệt → **http://localhost:5000**

#### 💻 **Cách 2: Command Line**

```bash
python detect_realtime.py
```

#### 🖥️ **Cách 3: Ứng Dụng Desktop (pywebview)**

```bash
python desktop_app.py
```

Cửa sổ ứng dụng mở trực tiếp, không cần thao tác với trình duyệt.

### Bước 3: Sử Dụng

- Bấm **"▶️ BẬT CAMERA"** (Web) hoặc script sẽ tự bắt đầu
- Hệ thống sẽ phát hiện ổ gà và cảnh báo
- Bấm **"Q"** để thoát (CLI) hoặc **"⏹️ TẮT CAMERA"** (Web)

---


**Cách chạy:**

```bash
python app.py
```

**Output:**

```
✅ Server running at: http://127.0.0.1:5000/
📖 Open your browser and go to: http://localhost:5000/
```

**Tính năng Web UI:**

- 📷 Xem video từ webcam trực tiếp
- 🔔 Cảnh báo tự động hiển thị
- 📊 Thống kê: khung hình, số lần phát hiện
- ⚙️ Điều chỉnh độ tin cậy (0.3-0.95)
- 🎯 Danh sách phát hiện gần đây

---


**Phím tắt:**

- `Q`: Thoát
- `P`: Tạm dừng/tiếp tục

---

## 🎯 Độ Tin Cậy (Confidence) - Hướng Dẫn

**Confidence** là mức độ chắc chắn để phát hiện ổ gà (0.0 - 1.0):

| Giá Trị     | Mô Tả                     | Khi Nào Dùng          |
| ----------- | ------------------------- | --------------------- |
| **0.3-0.4** | Rất nhạy, nhiều cảnh báo  | Vùng rủi ro cao       |
| **0.5**     | ✓ Cân bằng (mặc định)     | **Bình thường**       |
| **0.6-0.7** | Chặt chẽ, ít cảnh báo sai | Kiểm tra vùng an toàn |
| **0.8+**    | Cực chặt, rất ít cảnh báo | Debug/testing         |

---

## 🔧 Cải Thiện Hiệu Suất

### ⚡ Tăng Tốc Độ

1. Sử dụng GPU (nếu có NVIDIA)
2. Giảm resolution input
3. Dùng mô hình nhỏ (yolo11n)

### 🎯 Tăng Độ Chính Xác

1. Điều chỉnh confidence (0.5-0.6 là tốt)
2. Đảm bảo ánh sáng tốt
3. Làm sạch lens camera
4. Train lại mô hình với data đa dạng

### 🔇 Giảm False Positive (Cảnh báo Sai)

1. Tăng confidence lên 0.7+
2. Huấn luyện lại mô hình
3. Cải thiện chất lượng video

---

## ⚠️ Khắc Phục Sự Cố

### ❌ "Không tìm thấy file best.pt"

```bash
# Mô hình phải ở trong: runs/detect/*/weights/best.pt
# Kiểm tra xem folder runs/detect/ có tồn tại không
```

### ❌ "Cannot open webcam"

```bash
# 1. Kiểm tra webcam có kết nối không
# 2. Đóng ứng dụng khác dùng camera (Zoom, Teams...)
# 3. Thử lại
```

### ❌ "Port 5000 already in use"

```bash
# Sửa app.py dòng cuối:
app.run(debug=False, host='127.0.0.1', port=8080)  # Dùng port 8080
```

### ❌ "CUDA out of memory"

```bash
# Thêm vào đầu script:
# model = YOLO('best.pt').to('cpu')  # Dùng CPU thay GPU
```

### ❌ Không có âm thanh cảnh báo

```bash
pip install pygame
```

---

## 📁 Cấu Trúc Dự Án

```
Phat-hien-o-ga-tren-duong/
├── 🌐 app.py                        # Flask web app (KHUYẾN NGHỊ)
├── 💻 detect_realtime.py            # CLI script phát hiện real-time
├── 📷 detect_images.py              # Phát hiện ảnh tĩnh
├── 🎬 detect_video.py               # Phát hiện video
├── 🎯 pothole_segmentation_alert.py # Phát hiện + segmentation
├── ✓ check_environment.py           # Kiểm tra môi trường
├── 📊 visualize_labels.py           # Visualize dataset
│
├── 📦 requirements.txt              # Dependencies
├── 📖 README.md                     # Hướng dẫn này
│
├── templates/
│   └── 🌐 index.html               # Giao diện Web
│
├── runs/detect/                    # Thư mục mô hình
│   └── train/weights/best.pt      # Mô hình YOLO đã train
│
├── test/                           # Dữ liệu test
│   ├── images/                     # Ảnh test
│   └── labels/                     # Label
│
├── train/                          # Dữ liệu training
├── valid/                          # Dữ liệu validation
│
└── data.yaml                       # Dataset config
```

---

## 💡 Các Tình Huống Sử Dụng

### 📱 **Kiểm Tra Đường Tại Nhà**

```bash
python app.py
# Mở browser → http://localhost:5000
# Đặt webcam hướng cửa sổ
```

### **Test Trên Video Sẵn Có**

```bash
python detect_video.py --source test2.mp4
```
### Test real-time 
```bash
python detect_video.py --source 0
```

### 📡 **Chạy Trên Điện Thoại Cùng Mạng**

```bash
# Sửa app.py:
app.run(debug=False, host='0.0.0.0', port=5000)

# Trên điện thoại truy cập:
# http://<IP_máy_tính>:5000
```


---

## 🖥️ Đóng Gói Ứng Dụng Desktop (pywebview)

`desktop_app.py` đã tích hợp pywebview để mở giao diện giống ứng dụng Windows thực thụ.

### 1. Cài phụ thuộc

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Chạy thử app desktop

```bash
python desktop_app.py
```

- Flask + YOLO chạy trên background thread.
- pywebview hiển thị cửa sổ trỏ tới `http://127.0.0.1:5000/`.
- Đóng cửa sổ là tắt toàn bộ tiến trình.

### 3. Build bản phát hành bằng PyInstaller

```bash
# (khuyến nghị) Xoá build cũ để tránh lỗi file bị khoá
python scripts/cleanup_build.py

pyinstaller --noconfirm --clean pothole_app.spec
```

- Output: `dist/pothole_app/` (onedir). Copy hoặc nén toàn bộ thư mục để phát hành.
- Muốn 1 file duy nhất → mở `pothole_app.spec`, thêm `onefile=True` trong phần `EXE(...)`.

### 4. Dọn dẹp sau khi build

- Script nhanh: `python scripts/cleanup_build.py`

```powershell
Remove-Item -Recurse -Force build, dist
```

Script còn xoá `__pycache__`, giữ repo gọn gàng và tránh commit nhầm các file build.


## 🎓 Kiến Thức Kỹ Thuật

### Công Nghệ Sử Dụng

- **Model AI:** YOLOv8 (Ultralytics)
- **Framework:** PyTorch
- **Web Backend:** Flask + Python
- **Video Processing:** OpenCV
- **Audio:** Pygame Mixer

---


## Chạy dự án

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Chạy (chọn một)
python app.py                    # Web UI (khuyến nghị)
# hoặc
python detect_video.py        # CLI

# 3. Mở browser (nếu dùng Web UI)
# http://localhost:5000
```