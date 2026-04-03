"""Eye Aspect Ratio (EAR) based drowsiness detection."""

import numpy as np
from scipy.spatial import distance as dist

import config


class EyeDetector:
    def __init__(self):
        self.counter = 0

    def _eye_aspect_ratio(self, eye_points):
        """Compute EAR for 6 eye landmark points."""
        vertical_1 = dist.euclidean(eye_points[1], eye_points[5])
        vertical_2 = dist.euclidean(eye_points[2], eye_points[4])
        horizontal = dist.euclidean(eye_points[0], eye_points[3])
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def detect(self, landmarks):
        """Detect drowsiness from face landmarks.

        Returns:
            (ear_value, is_drowsy)
        """
        left_eye = landmarks[config.LEFT_EYE]
        right_eye = landmarks[config.RIGHT_EYE]

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < config.EAR_THRESHOLD:
            self.counter += 1
        else:
            self.counter = 0

        is_drowsy = self.counter >= config.EAR_CONSECUTIVE_FRAMES
        return avg_ear, is_drowsy
