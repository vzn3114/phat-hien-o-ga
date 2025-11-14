"""
Script kiểm tra môi trường và các thư viện cần thiết
"""
import sys

def check_python_version():
    """Kiểm tra phiên bản Python"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Cần Python 3.10+")
        return False

def check_library(lib_name, import_name=None):
    """Kiểm tra thư viện đã cài đặt chưa"""
    if import_name is None:
        import_name = lib_name
    
    try:
        __import__(import_name)
        print(f"✅ {lib_name} - Đã cài đặt")
        return True
    except ImportError:
        print(f"❌ {lib_name} - Chưa cài đặt. Chạy: pip install {lib_name}")
        return False

def check_files():
    """Kiểm tra các file cần thiết"""
    import os
    import glob
    
    print("\n📁 Kiểm tra file:")
    
    # Kiểm tra mô hình detect
    weight_paths = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
    if weight_paths:
        latest = max(weight_paths, key=os.path.getmtime)
        print(f"✅ Mô hình detect: {latest}")
    else:
        print("❌ Không tìm thấy mô hình detect (best.pt)")
    
    # Kiểm tra mô hình segment
    seg_paths = glob.glob("runs/segment/**/weights/best.pt", recursive=True)
    if seg_paths:
        latest = max(seg_paths, key=os.path.getmtime)
        print(f"✅ Mô hình segment: {latest}")
    else:
        print("⚠️  Không tìm thấy mô hình segment (cần cho pothole_segmentation_alert.py)")
    
    # Kiểm tra file cảnh báo
    if os.path.exists("canhbao.mp3"):
        print("✅ File cảnh báo: canhbao.mp3")
    else:
        print("⚠️  File canhbao.mp3 chưa có (sẽ tự động tạo khi chạy script)")
    
    # Kiểm tra file test
    if os.path.exists("anhtest.jpg"):
        print("✅ File ảnh test: anhtest.jpg")
    else:
        print("⚠️  Không tìm thấy anhtest.jpg")
    
    if os.path.exists("test2.mp4"):
        print("✅ File video test: test2.mp4")
    else:
        print("⚠️  Không tìm thấy test2.mp4")

def check_gpu():
    """Kiểm tra GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)} - Sẵn sàng")
            return True
        else:
            print("⚠️  GPU: Không có (sẽ dùng CPU)")
            return False
    except ImportError:
        print("⚠️  Không thể kiểm tra GPU (torch chưa cài đặt)")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🔍 KIỂM TRA MÔI TRƯỜNG DỰ ÁN PHÁT HIỆN Ổ GÀ")
    print("=" * 50)
    
    print("\n🐍 Kiểm tra Python:")
    python_ok = check_python_version()
    
    print("\n📦 Kiểm tra thư viện:")
    libs_ok = True
    libs_ok &= check_library("opencv-python", "cv2")
    libs_ok &= check_library("ultralytics", "ultralytics")
    libs_ok &= check_library("gtts", "gtts")
    libs_ok &= check_library("playsound", "playsound")
    libs_ok &= check_library("numpy", "numpy")
    libs_ok &= check_library("torch", "torch")
    
    check_files()
    
    print("\n🖥️  Kiểm tra GPU:")
    check_gpu()
    
    print("\n" + "=" * 50)
    if python_ok and libs_ok:
        print("✅ Môi trường đã sẵn sàng! Bạn có thể chạy các script.")
        print("\n🚀 Cách chạy:")
        print("   - Phát hiện ảnh: python detect_images.py")
        print("   - Phát hiện video: python detect_video.py")
        print("   - Segmentation: python pothole_segmentation_alert.py")
    else:
        print("❌ Cần cài đặt thêm một số thư viện.")
        print("   Chạy: pip install -r requirements.txt")
    print("=" * 50)


