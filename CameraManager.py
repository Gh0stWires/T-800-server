from picamera2 import Picamera2
import cv2
import threading
import time

class CameraManager:
    """Singleton class to manage Picamera2 instance and provide a thread-safe frame buffer."""
    
    _instance = None  # Static variable to hold the singleton instance
    _lock = threading.Lock()  # Thread lock for safety

    def __new__(cls):
        """Ensures only one instance of CameraManager is created."""
        if cls._instance is None:
            with cls._lock:  # Double-checked locking for thread safety
                if cls._instance is None:
                    cls._instance = super(CameraManager, cls).__new__(cls)
                    cls._instance._init_camera()
        return cls._instance

    def _init_camera(self):
        """Initializes the camera and starts frame capture in a separate thread."""
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_video_configuration(main={"size": (640, 480)}))
        self.camera.start()
        
        self.frame_buffer = None
        self.lock = threading.Lock()
        
        # Start the frame capture thread
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        """Continuously capture frames and update the shared buffer."""
        while True:
            with self.lock:
                frame = self.camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', frame)
                self.frame_buffer = buffer.tobytes()
            #time.sleep(0.05) 

    def get_frame(self):
        """Returns the latest frame safely from the buffer."""
        with self.lock:
            return self.frame_buffer

# ✅ Ensure only one CameraManager instance
camera_manager = CameraManager()
