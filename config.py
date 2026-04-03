"""Central configuration for all thresholds and constants."""

# --- Eye Aspect Ratio (Drowsiness) ---
EAR_THRESHOLD = 0.22
EAR_CONSECUTIVE_FRAMES = 20  # ~0.7s at 30fps

# MediaPipe FaceMesh eye landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# --- Mouth Aspect Ratio (Yawning) ---
YAWN_THRESHOLD = 0.6
YAWN_CONSECUTIVE_FRAMES = 15

# MediaPipe FaceMesh mouth landmark indices
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# --- Head Pose (Distraction) ---
YAW_THRESHOLD = 30      # degrees
PITCH_THRESHOLD = 25     # degrees
HEAD_POSE_CONSECUTIVE_FRAMES = 15

# 6 key landmark indices for solvePnP
HEAD_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]

# 3D model points (generic face model)
MODEL_POINTS_3D = [
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
]

# --- Phone Detection ---
PHONE_CONFIDENCE = 0.5
PHONE_CLASS_ID = 67  # COCO class for "cell phone"
PHONE_DETECTION_WINDOW = 5   # rolling window size
PHONE_MIN_DETECTIONS = 3     # min detections in window to trigger

# --- Seatbelt Detection ---
SEATBELT_CONFIDENCE = 0.5

# --- Alert Configuration ---
ALERTS = {
    "drowsiness": {
        "sound": "alerts/sounds/drowsy_alert.wav",
        "color": (0, 0, 255),        # Red (BGR)
        "message": "DROWSINESS DETECTED - WAKE UP!",
        "cooldown": 5.0,
        "priority": 5,
    },
    "distraction": {
        "sound": "alerts/sounds/distraction_alert.wav",
        "color": (0, 255, 255),      # Yellow (BGR)
        "message": "EYES ON THE ROAD!",
        "cooldown": 4.0,
        "priority": 4,
    },
    "phone": {
        "sound": "alerts/sounds/phone_alert.wav",
        "color": (255, 0, 0),        # Blue (BGR)
        "message": "PHONE USAGE DETECTED!",
        "cooldown": 6.0,
        "priority": 3,
    },
    "yawning": {
        "sound": "alerts/sounds/yawn_alert.wav",
        "color": (0, 165, 255),      # Orange (BGR)
        "message": "YAWNING DETECTED - Stay Alert!",
        "cooldown": 8.0,
        "priority": 2,
    },
    "seatbelt": {
        "sound": "alerts/sounds/seatbelt_alert.wav",
        "color": (0, 0, 200),        # Dark red (BGR)
        "message": "PLEASE WEAR YOUR SEATBELT!",
        "cooldown": 15.0,
        "priority": 1,
    },
}

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
