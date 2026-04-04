"""Phone detection using YOLOv8 in a background thread."""

import threading
from collections import deque

from ultralytics import YOLO

import config


class PhoneDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self._detected = False
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._latest_frame = None
        self._frame_event = threading.Event()
        self._history = deque(maxlen=config.PHONE_DETECTION_WINDOW)

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

            results = self.model(
                frame,
                conf=config.PHONE_CONFIDENCE,
                classes=[config.PHONE_CLASS_ID],
                verbose=False,
            )

            found = len(results[0].boxes) > 0
            with self._lock:
                self._history.append(found)
                self._detected = sum(self._history) >= config.PHONE_MIN_DETECTIONS

    @property
    def is_phone_detected(self):
        with self._lock:
            return self._detected

    def stop(self):
        self._running = False
        self._frame_event.set()
        if self._thread:
            self._thread.join(timeout=2)
