from flask import Flask, Response
from CameraManager import camera_manager
import os 

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get absolute directory
cert_path = os.path.join(BASE_DIR, "cert.pem")
key_path = os.path.join(BASE_DIR, "key.pem")

def generate_frames():
    """Yield the latest frame from the singleton camera instance."""
    while True:
        frame = camera_manager.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stream')
def stream():
    """Serve the MJPEG camera stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, ssl_context=(cert_path, key_path), debug=False, threaded=True)
