import threading
import time
import urllib.request

import webview

from app import app, load_model, init_audio, load_mp3_sound

HOST = "127.0.0.1"
PORT = 5000


def start_flask_server():
    """Start Flask server with model/audio initialization."""
    if not load_model():
        raise RuntimeError("Không thể tải mô hình YOLO (best.pt).")
    
    init_audio()
    load_mp3_sound()
    
    app.run(
        debug=False,
        host=HOST,
        port=PORT,
        threaded=True,
        use_reloader=False
    )


def wait_for_server(timeout=20):
    """Wait until Flask server responds."""
    endpoint = f"http://{HOST}:{PORT}/api/status"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(endpoint) as resp:
                if resp.status in (200, 202):
                    return True
        except Exception:
            time.sleep(0.3)
    
    return False


def main():
    server_thread = threading.Thread(target=start_flask_server, daemon=True)
    server_thread.start()
    
    if not wait_for_server():
        raise RuntimeError("Server không khởi động được trong 20 giây.")
    
    window = webview.create_window(
        "Phát Hiện Ổ Gà",
        f"http://{HOST}:{PORT}/",
        width=1280,
        height=800,
        resizable=True,
        text_select=True
    )
    
    webview.start()
    # Khi cửa sổ đóng, chương trình kết thúc và thread daemon cũng dừng.


if __name__ == "__main__":
    main()

