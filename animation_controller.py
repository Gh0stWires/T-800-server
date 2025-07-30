import threading
import time
import random
from robot_hat import Servo, PWM

# Servo setup
pwm_eye_lr = PWM("P0")
pwm_eye_ud = PWM("P1")
pwm_neck = PWM("P2")

servo_eye_lr = Servo(pwm_eye_lr)
servo_eye_ud = Servo(pwm_eye_ud)
servo_neck = Servo(pwm_neck)

# Center positions
EYE_CENTER_LR = -70
EYE_CENTER_UD = 55
NECK_CENTER = 0

# Offsets
EYE_OFFSET_LR = 20   # eye left-right
EYE_OFFSET_UD = 20    # eye up-down
NECK_OFFSET = 40 

# Internal flag for stopping the animation
_thinking = False
_thread = None

def smooth_move(servo, start, end, steps=8, delay=0.03):
    step_size = (end - start) / steps
    for i in range(steps + 1):
        servo.angle(start + i * step_size)
        time.sleep(delay)

def _run_thinking_loop():
    global _thinking

    current_eye_lr = EYE_CENTER_LR
    current_eye_ud = EYE_CENTER_UD
    current_neck = NECK_CENTER

    while _thinking:
        # Random targets
        target_eye_lr = EYE_CENTER_LR + random.uniform(-EYE_OFFSET_LR, EYE_OFFSET_LR)
        target_eye_ud = EYE_CENTER_UD + random.uniform(-EYE_OFFSET_UD, EYE_OFFSET_UD)
        target_neck = NECK_CENTER + random.uniform(-NECK_OFFSET, NECK_OFFSET)

        # Smooth movements
        smooth_move(servo_eye_lr, current_eye_lr, target_eye_lr)
        smooth_move(servo_eye_ud, current_eye_ud, target_eye_ud)
        smooth_move(servo_neck, current_neck, target_neck)

        current_eye_lr = target_eye_lr
        current_eye_ud = target_eye_ud
        current_neck = target_neck

        time.sleep(random.uniform(0.6, 1.5))

    # Return to center
    smooth_move(servo_eye_lr, current_eye_lr, EYE_CENTER_LR)
    smooth_move(servo_eye_ud, current_eye_ud, EYE_CENTER_UD)
    smooth_move(servo_neck, current_neck, NECK_CENTER)

def start_thinking_animation():
    global _thinking, _thread
    if not _thinking:
        _thinking = True
        _thread = threading.Thread(target=_run_thinking_loop, daemon=True)
        _thread.start()

def stop_thinking_animation():
    global _thinking, _thread
    _thinking = False

    # Wait for the thread to finish (if running)
    if _thread is not None:
        _thread.join()
        _thread = None

    # Force recenter in case the thread didn't complete
    servo_eye_lr.angle(EYE_CENTER_LR)
    servo_eye_ud.angle(EYE_CENTER_UD)
    servo_neck.angle(NECK_CENTER)


