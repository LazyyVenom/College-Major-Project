"""Seatbelt detection using YOLOv8 in a background thread.

Uses a pretrained seatbelt detection model. If custom weights are not available,
falls back to a heuristic-based approach using edge detection.
"""

import os
import threading
import time

import cv2
import numpy as np

import config

CUSTOM_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "models", "seatbelt.pt")


class SeatbeltDetector:
    def __init__(self):
        self._is_wearing = True  # Default to True (assume wearing until proven otherwise)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._latest_frame = None
        self._use_yolo = os.path.exists(CUSTOM_WEIGHTS)

        if self._use_yolo:
            from ultralytics import YOLO
            self.model = YOLO(CUSTOM_WEIGHTS)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update_frame(self, frame):
        with self._lock:
            self._latest_frame = frame.copy()

    def _run(self):
        while self._running:
            with self._lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.05)
                continue

            if self._use_yolo:
                wearing = self._detect_yolo(frame)
            else:
                wearing = self._detect_heuristic(frame)

            with self._lock:
                self._is_wearing = wearing

            time.sleep(0.1)  # Don't need to check every frame

    def _detect_yolo(self, frame):
        """Detect seatbelt using custom YOLO model."""
        results = self.model(frame, conf=config.SEATBELT_CONFIDENCE, verbose=False)
        # Assumes model has classes: 0=seatbelt (present)
        return len(results[0].boxes) > 0

    def _detect_heuristic(self, frame):
        """Fallback: detect diagonal strap using edge detection.

        Looks for a strong diagonal line in the chest region (left shoulder to right hip).
        This is a rough heuristic and less reliable than a trained model.
        """
        h, w = frame.shape[:2]
        # Region of interest: upper-left chest area (where seatbelt crosses)
        roi = frame[int(h * 0.2):int(h * 0.7), int(w * 0.1):int(w * 0.5)]

        if roi.size == 0:
            return True

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Look for diagonal lines (seatbelt strap)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=80, maxLineGap=10)

        if lines is None:
            return False

        # Check for lines at roughly 30-60 degree angles (diagonal strap)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if 30 <= angle <= 60:
                return True

        return False

    @property
    def is_wearing(self):
        with self._lock:
            return self._is_wearing

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
