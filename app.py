"""
Flask Web App for Real-time Pothole Detection
Ứng dụng web phát hiện ổ gà theo thời gian thực
"""
import cv2
import glob
import os
import threading
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, request
import base64
from io import BytesIO
import pygame
import time
from pothole_filter import PotholeFilter
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# ==============================
# Configuration
# ==============================
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ==============================
# Global Variables
# ==============================
detection_active = False
webcam_source = 0  # Default: built-in webcam
confidence_threshold = 0.5
model = None
latest_detections = []
frame_count = 0
detection_enabled = True
audio_mode = "beep"  # Options: "beep", "mp3", "off"
sound_object = None  # Store pygame sound object

# ==============================
# Initialize Model
# ==============================
def load_model():
    global model
    try:
        weight_paths = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
        if not weight_paths:
            raise FileNotFoundError("❌ Không tìm thấy file best.pt")
        
        latest_weight = max(weight_paths, key=os.path.getmtime)
        model = YOLO(latest_weight)
        print(f"✅ Model loaded: {latest_weight}")
        return True
    except Exception as e:
        print(f"❌ Load model error: {e}")
        return False

# ==============================
# Audio Alert (Using Pygame)
# ==============================
def init_audio():
    """Initialize pygame mixer for sound"""
    try:
        pygame.mixer.init()
        print("✅ Audio mixer initialized")
        return True
    except Exception as e:
        print(f"⚠️  Audio initialization failed: {e}")
        return False

def load_mp3_sound():
    """Load MP3 alert sound"""
    global sound_object
    try:
        if os.path.exists("canhbao.mp3"):
            sound_object = pygame.mixer.Sound("canhbao.mp3")
            print("✅ MP3 sound loaded: canhbao.mp3")
            return True
        else:
            print("⚠️  canhbao.mp3 not found")
            return False
    except Exception as e:
        print(f"⚠️  Load MP3 error: {e}")
        return False

def play_alert():
    """Play alert sound based on audio_mode"""
    global audio_mode, sound_object
    
    if audio_mode == "off":
        return  # No sound
    
    try:
        if audio_mode == "beep":
            # Generate beep sound using frequency
            sample_rate = 44100
            duration = 0.5  # seconds
            frequency = 880  # Hz
            
            frames = int(sample_rate * duration)
            arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
            arr = (arr * 32767).astype(np.int16)
            
            # Create stereo sound
            stereo = np.zeros((frames, 2), dtype=np.int16)
            stereo[:, 0] = arr
            stereo[:, 1] = arr
            
            sound = pygame.sndarray.make_sound(stereo)
            sound.play()
            print("🔔 Beep alert played")
        
        elif audio_mode == "mp3":
            # Play MP3 file
            if sound_object is not None:
                sound_object.play()
                print("🔔 MP3 alert played")
            else:
                print("⚠️  MP3 sound not loaded, playing beep instead")
                play_alert_beep()
    
    except Exception as e:
        print(f"⚠️  Play alert error: {e}")

def play_alert_beep():
    """Helper function to play beep sound"""
    try:
        sample_rate = 44100
        duration = 0.5
        frequency = 880
        
        frames = int(sample_rate * duration)
        arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
        arr = (arr * 32767).astype(np.int16)
        
        stereo = np.zeros((frames, 2), dtype=np.int16)
        stereo[:, 0] = arr
        stereo[:, 1] = arr
        
        sound = pygame.sndarray.make_sound(stereo)
        sound.play()
    except Exception as e:
        print(f"⚠️  Beep error: {e}")

# ==============================
# Detection Thread
# ==============================
def preprocess_frame(frame):
    """Cải thiện ảnh để phát hiện tốt hơn"""
    # Tăng contrast & brightness
    alpha = 1.2  # Contrast
    beta = 30    # Brightness
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Adaptive histogram equalization
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame = cv2.merge([l, a, b])
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
    
    return frame

def detection_loop():
    """Background detection thread - IMPROVED"""
    global detection_active, frame_count, latest_detections
    
    if model is None:
        print("❌ Model not loaded")
        return
    
    cap = cv2.VideoCapture(webcam_source)
    if not cap.isOpened():
        print(f"❌ Cannot open webcam source: {webcam_source}")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    
    print("🎥 Webcam opened (1280x720), starting detection...")
    last_alert_time = 0
    detection_history = []  # Track detections
    
    while detection_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        if not detection_enabled:
            continue
        
        try:
            # Preprocess frame for better detection
            frame_processed = preprocess_frame(frame)
            
            # Run inference
            results = model(frame_processed, verbose=False, conf=confidence_threshold)
            
            detected = False
            detections_list = []
            
            for result in results:
                if hasattr(result, "boxes") and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    # Apply advanced filtering to remove false positives
                    filtered_boxes, filtered_confs, filter_reasons = PotholeFilter.filter_detections(
                        boxes, confs, display_frame.shape, min_confidence=0.45
                    )
                    
                    for box, conf in zip(filtered_boxes, filtered_confs):
                        detected = True
                        x1, y1, x2, y2 = map(int, box)
                        
                        detections_list.append({
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "confidence": float(conf)
                        })
                        
                        # Draw rectangle on display frame (HD quality)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(display_frame, f"POTHOLE {conf:.2f}",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.8, (0, 255, 0), 2)
            
            # Alert if pothole detected
            if detected and (time.time() - last_alert_time > 3):
                play_alert()
                print(f"⚠️  ALERT: {len(detections_list)} pothole(s) detected!")
                last_alert_time = time.time()
            
            # Resize for web display (reduce bandwidth)
            display_small = cv2.resize(display_frame, (1024, 576))
            
            # Add info overlay
            info_text = f"Frame: {frame_count} | Detections: {len(detections_list)}"
            cv2.putText(display_small, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode frame to base64 for web display
            _, buffer = cv2.imencode('.jpg', display_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode()
            
            latest_detections = {
                "frame": frame_b64,
                "detections": detections_list,
                "frame_count": frame_count,
                "timestamp": time.time()
            }
            
            # Keep detection history (last 100 frames)
            detection_history.append({
                "frame": frame_count,
                "count": len(detections_list),
                "time": time.time()
            })
            if len(detection_history) > 100:
                detection_history.pop(0)
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            continue
    
    cap.release()
    print("🛑 Detection stopped")

# ==============================
# Flask Routes
# ==============================
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start webcam detection"""
    global detection_active
    
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 400
    
    if detection_active:
        return jsonify({"status": "warning", "message": "Detection already running"}), 200
    
    detection_active = True
    thread = threading.Thread(target=detection_loop, daemon=True)
    thread.start()
    
    return jsonify({"status": "success", "message": "Detection started"}), 200

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global detection_active
    detection_active = False
    return jsonify({"status": "success", "message": "Detection stopped"}), 200

@app.route('/api/frame')
def get_frame():
    """Get current frame"""
    if not latest_detections:
        return jsonify({"status": "waiting", "message": "No frame yet"}), 202
    
    return jsonify({
        "status": "success",
        "frame": latest_detections.get("frame", ""),
        "detections": latest_detections.get("detections", []),
        "frame_count": latest_detections.get("frame_count", 0)
    }), 200

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get/set detection settings"""
    global confidence_threshold, detection_enabled, audio_mode
    
    if request.method == 'POST':
        data = request.get_json()
        if 'confidence' in data:
            confidence_threshold = float(data['confidence'])
        if 'enabled' in data:
            detection_enabled = bool(data['enabled'])
        if 'audio_mode' in data:
            new_mode = data['audio_mode']
            if new_mode in ["beep", "mp3", "off"]:
                audio_mode = new_mode
                print(f"🔊 Audio mode changed to: {audio_mode}")
                return jsonify({
                    "status": "success",
                    "message": f"Audio mode changed to {audio_mode}"
                }), 200
            else:
                return jsonify({
                    "status": "error",
                    "message": "Invalid audio mode"
                }), 400
        return jsonify({"status": "success"}), 200
    
    return jsonify({
        "confidence_threshold": confidence_threshold,
        "detection_enabled": detection_enabled,
        "audio_mode": audio_mode,
        "frame_count": frame_count
    }), 200

@app.route('/api/status')
def status():
    """Get system status"""
    return jsonify({
        "detection_active": detection_active,
        "model_loaded": model is not None,
        "frame_count": frame_count,
        "confidence_threshold": confidence_threshold
    }), 200

# ==============================
# File Upload & Detection
# ==============================

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def detect_in_image(image_path, min_confidence=0.45):
    """Detect potholes in a single image"""
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None, []
        
        # Preprocess frame
        frame_processed = preprocess_frame(frame)
        
        # Run inference
        results = model(frame_processed, verbose=False, conf=confidence_threshold)
        
        display_frame = frame.copy()
        detections_list = []
        
        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                # Apply filter
                filtered_boxes, filtered_confs, _ = PotholeFilter.filter_detections(
                    boxes, confs, display_frame.shape, min_confidence=0.45
                )
                
                for box, conf in zip(filtered_boxes, filtered_confs):
                    x1, y1, x2, y2 = map(int, box)
                    detections_list.append({
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "confidence": float(conf)
                    })
                    
                    # Draw rectangle
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(display_frame, f"POTHOLE {conf:.2f}",
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.8, (0, 255, 0), 2)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode()
        
        return frame_b64, detections_list
    
    except Exception as e:
        print(f"❌ Image detection error: {e}")
        return None, []

def detect_in_video(video_path, min_confidence=0.45):
    """Detect potholes in video and return frames"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        detection_frames = []
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to speed up (process every 5 frames)
            if frame_count % 5 != 0:
                continue
            
            # Preprocess frame
            frame_processed = preprocess_frame(frame)
            
            # Run inference
            results = model(frame_processed, verbose=False, conf=confidence_threshold)
            
            display_frame = frame.copy()
            detections_list = []
            
            for result in results:
                if hasattr(result, "boxes") and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    # Apply filter
                    filtered_boxes, filtered_confs, _ = PotholeFilter.filter_detections(
                        boxes, confs, display_frame.shape, min_confidence=0.45
                    )
                    
                    for box, conf in zip(filtered_boxes, filtered_confs):
                        x1, y1, x2, y2 = map(int, box)
                        detections_list.append({
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "confidence": float(conf)
                        })
                        
                        # Draw rectangle
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(display_frame, f"POTHOLE {conf:.2f}",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.8, (0, 255, 0), 2)
            
            # Add frame info
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode()
            
            detection_frames.append({
                "frame": frame_b64,
                "frame_number": frame_count,
                "detections": detections_list
            })
            
            all_detections.extend(detections_list)
            
            # Limit frames
            if len(detection_frames) >= 100:
                break
        
        cap.release()
        
        return {
            "frames": detection_frames,
            "total_detections": len(all_detections),
            "total_frames": frame_count,
            "fps": fps
        }
    
    except Exception as e:
        print(f"❌ Video detection error: {e}")
        return None

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Detect potholes in uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400
        
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({"status": "error", "message": "Invalid image format"}), 400
        
        if model is None:
            return jsonify({"status": "error", "message": "Model not loaded"}), 500
        
        # Save uploaded file
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect
        frame_b64, detections = detect_in_image(filepath, min_confidence=confidence_threshold)
        
        if frame_b64 is None:
            return jsonify({"status": "error", "message": "Failed to process image"}), 400
        
        # Play alert if detections found
        if len(detections) > 0:
            play_alert()
        
        return jsonify({
            "status": "success",
            "frame": frame_b64,
            "detections": detections,
            "detection_count": len(detections)
        }), 200
    
    except Exception as e:
        print(f"❌ Image upload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    """Detect potholes in uploaded video"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400
        
        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({"status": "error", "message": "Invalid video format"}), 400
        
        if model is None:
            return jsonify({"status": "error", "message": "Model not loaded"}), 500
        
        # Save uploaded file
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"🎬 Processing video: {filename}")
        
        # Detect
        result = detect_in_video(filepath, min_confidence=confidence_threshold)
        
        if result is None:
            return jsonify({"status": "error", "message": "Failed to process video"}), 400
        
        # Play alert if detections found
        if result["total_detections"] > 0:
            play_alert()
        
        return jsonify({
            "status": "success",
            "frames": result["frames"],
            "total_detections": result["total_detections"],
            "total_frames": result["total_frames"],
            "fps": result["fps"]
        }), 200
    
    except Exception as e:
        print(f"❌ Video upload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500

# ==============================
# Main
# ==============================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Pothole Detection Web App - Starting...")
    print("=" * 60)
    
    # Load model
    if not load_model():
        print("❌ Failed to load model. Exiting.")
        exit(1)
    
    # Init audio
    init_audio()
    load_mp3_sound()
    
    # Run Flask app
    print("\n Server running at: http://127.0.0.1:5000/")
    print("📖 Open your browser and go to: http://localhost:5000/")
    print("\nPress Ctrl+C to stop server")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
