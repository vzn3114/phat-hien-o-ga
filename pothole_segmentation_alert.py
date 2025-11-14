# pothole_segmentation_alert.py
import cv2
import glob
import os
import numpy as np
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

# Load YOLO model
model = YOLO(latest_weight)

# ==============================
# 3. Mở video
# ==============================
video_path = 0   # đổi đường dẫn nếu cần
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"⚠️ Không thể mở video {video_path}")

last_alert_time = 0

# ==============================
# 4. Chạy detect + cảnh báo
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detected = False

    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                if conf > 0.5:  # Ngưỡng confidence 50%
                    detected = True
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Vẽ bounding box (màu xanh lá)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Vẽ label
                    label = f"O GA {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Nếu phát hiện ổ gà thì cảnh báo
    if detected and (time.time() - last_alert_time > 5):
        print("⚠️ Cảnh báo: Phía trước có ổ gà!")
        cv2.putText(frame, "CANH BAO: O GA!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Phát âm thanh cảnh báo (nếu playsound3 không hoạt động, chỉ in log)
        try:
            playsound("canhbao.mp3", block=False)
        except Exception as e:
            print(f"⚠️ Lỗi phát âm thanh: {e}")
        
        last_alert_time = time.time()

    # Hiển thị kết quả
    cv2.imshow("Pothole Detection + Alert", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
