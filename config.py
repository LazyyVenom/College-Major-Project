"""Central configuration for all thresholds and constants."""

# --- Eye State Detection (Drowsiness) ---
EYE_SCORE_THRESHOLD = 0.22
EYE_SCORE_CONSECUTIVE_FRAMES = 20  # ~0.7s at 30fps

# MediaPipe FaceMesh eye landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# --- Yawn Detection ---
YAWN_THRESHOLD = 0.6
YAWN_CONSECUTIVE_FRAMES = 15

# MediaPipe FaceMesh mouth landmark indices
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# --- Head Pose (Distraction) ---
YAW_THRESHOLD = 40      # degrees
PITCH_THRESHOLD = 40     # degrees
HEAD_POSE_CONSECUTIVE_FRAMES = 30

# 6 key landmark indices for solvePnP
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

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
SEATBELT_ENABLED = False  # Set to True to enable seatbelt detection
SEATBELT_CONFIDENCE = 0.5

# --- Drunk / Impairment Detection ---
DRUNK_THRESHOLD = 0.55           # combined score threshold (0-1)
DRUNK_CONSECUTIVE_FRAMES = 35    # ~1.2s at 30fps before alerting
DRUNK_WINDOW_FRAMES = 160        # sliding window (~5.3s at 30fps)
DRUNK_MIN_FRAMES = 75            # minimum frames before scoring

# Landmark indices for asymmetry calculation
DRUNK_LEFT_MOUTH = 61
DRUNK_RIGHT_MOUTH = 291
DRUNK_NOSE_TIP = 1

# Per-signal thresholds
DRUNK_EYE_STD_THRESHOLD = 0.07       # eye score standard deviation
DRUNK_SWAY_THRESHOLD = 15.0          # head sway std (degrees)
DRUNK_ASYMMETRY_THRESHOLD = 0.18     # facial asymmetry
DRUNK_BLINK_STD_THRESHOLD = 10.0     # blink duration variance (frames)
DRUNK_CLOSURE_RATIO_THRESHOLD = 0.35 # fraction of frames with eyes closed

# Signal weights (sum to 1.0)
DRUNK_WEIGHT_EYE_VAR = 0.20
DRUNK_WEIGHT_SWAY = 0.30
DRUNK_WEIGHT_ASYMMETRY = 0.20
DRUNK_WEIGHT_BLINK = 0.15
DRUNK_WEIGHT_CLOSURE = 0.15

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
    "drunk": {
        "sound": "alerts/sounds/drunk_alert.wav",
        "color": (0, 50, 200),       # Dark orange (BGR)
        "message": "IMPAIRMENT DETECTED - PULL OVER!",
        "cooldown": 10.0,
        "priority": 6,
    },
}

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
