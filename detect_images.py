 # detect_images.py
import cv2
import os
import glob
import time
from ultralytics import YOLO
from gtts import gTTS
from playsound3 import playsound

# ==============================
# 1. Tạo file âm thanh cảnh báo nếu chưa có
# ==============================
if not os.path.exists("canhbao.mp3"):
    tts = gTTS("Cảnh báo! Phía trước có ổ gà, hãy giảm tốc độ!", lang="vi")
    tts.save("canhbao.mp3")
    print("✅ Đã tạo file canhbao.mp3")

# ==============================
# 2. Tìm file YOLO best.pt mới nhất
# ==============================
weight_paths = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
if not weight_paths:
    raise FileNotFoundError("⚠️ Không tìm thấy file best.pt trong runs/detect/")

latest_weight = max(weight_paths, key=os.path.getmtime)
print(f"✅ Đang dùng mô hình: {latest_weight}")

# Load YOLO
model = YOLO(latest_weight)

# ==============================
# 3. Đọc tất cả ảnh trong thư mục images/
# ==============================
image_paths = glob.glob("anhtest.jpg")  # đọc tất cả ảnh (jpg/png/...)
if not image_paths:
    raise FileNotFoundError("⚠️ Không tìm thấy ảnh trong thư mục images/")

# Tạo thư mục lưu kết quả
os.makedirs("results", exist_ok=True)

last_alert_time = 0   # để tránh cảnh báo liên tục

for img_path in image_paths:
    frame = cv2.imread(img_path)

    # Detect
    results = model(frame, verbose=False)

    detected = False
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        names = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, names):
            if conf > 0.5:  # ngưỡng tự tin
                detected = True
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "O GA!!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Nếu phát hiện ổ gà -> cảnh báo bằng âm thanh
    if detected and (time.time() - last_alert_time > 5):
        print(f"⚠️ Cảnh báo: Ổ gà trong ảnh {os.path.basename(img_path)}")
        playsound("canhbao.mp3", block=False)
        last_alert_time = time.time()

    # Hiển thị ảnh
    cv2.imshow("Pothole Detection", frame)

    # Lưu ảnh kết quả vào thư mục results/
    save_path = os.path.join("results", os.path.basename(img_path))
    cv2.imwrite(save_path, frame)
    print(f"💾 Đã lưu kết quả: {save_path}")

    cv2.waitKey(0)  # nhấn phím bất kỳ để qua ảnh tiếp theo

cv2.destroyAllWindows()
