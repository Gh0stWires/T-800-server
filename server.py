from flask import Flask, request, jsonify, Response, stream_with_context
from CameraManager import camera_manager
from tasks import process_chat_task 
from animation_controller import start_thinking_animation, stop_thinking_animation
from ai_processor import ask_t800
from ai import DEFAULT_AGENT_NAME, ask_ai
import json
import os
import requests

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get absolute directory
cert_path = os.path.join(BASE_DIR, "cert.pem")
key_path = os.path.join(BASE_DIR, "key.pem")

app = Flask(__name__)

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

@app.route("/speak", methods=["POST"])
def speak_proxy():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Forward to Windows Flask TTS server (no HTTPS needed)
        tts_response = requests.post(
            "http://10.0.0.145:5001/speak",  # your Windows server IP
            json={"text": text},
            timeout=120,
            stream=True
        )

        if tts_response.status_code != 200:
            return jsonify({"error": "TTS server error"}), 502

        return Response(
            tts_response.iter_content(chunk_size=4096),
            content_type="audio/wav"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = data.get("userId", "default_user")  # fallback just in case
    message = data.get("message", "")
    
    agent_data = data.get("agent", {})
    agent_name = agent_data.get("name", DEFAULT_AGENT_NAME)
    system_prompt = agent_data.get("systemPrompt", None)

    def generate():
        for event in ask_ai(user_id, message, agent_name, system_prompt):
            yield json.dumps(event) + "\n"
    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")



    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
