"""Seatbelt detection using YOLOv5 pretrained model in a background thread.

Uses pretrained weights from kaggle.com/datasets/sachinmlwala/seatbelt3.
Classes: 0 = "No Seatbelt", 1 = "Seatbelt Worn".
Falls back to heuristic edge detection if weights are not available.
"""

import os
import threading

import cv2
import numpy as np

import config

CUSTOM_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "models", "seatbelt.pt")

# Class IDs from sachinmlwala/seatbelt3 YOLOv5 model
CLASS_NO_SEATBELT = 0
CLASS_SEATBELT = 1


class SeatbeltDetector:
    def __init__(self):
        self._is_wearing = True
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._latest_frame = None
        self._frame_event = threading.Event()
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
        self._frame_event.set()

    def _run(self):
        while self._running:
            if not self._frame_event.wait(timeout=0.1):
                continue
            self._frame_event.clear()

            with self._lock:
                frame = self._latest_frame

            if frame is None:
                continue

            if self._use_yolo:
                wearing = self._detect_yolo(frame)
            else:
                wearing = self._detect_heuristic(frame)

            with self._lock:
                self._is_wearing = wearing

    def _detect_yolo(self, frame):
        """Detect seatbelt using pretrained YOLOv5 model.

        Model classes: 0 = No Seatbelt, 1 = Seatbelt Worn.
        Returns True if seatbelt is detected (class 1) or no detections at all.
        Returns False only if "No Seatbelt" (class 0) is detected.
        """
        results = self.model(frame, conf=config.SEATBELT_CONFIDENCE, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return True  # No detection, assume wearing

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == CLASS_NO_SEATBELT:
                return False

        return True

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
        self._frame_event.set()
        if self._thread:
            self._thread.join(timeout=2)
