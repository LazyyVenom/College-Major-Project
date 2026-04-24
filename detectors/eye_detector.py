"""CNN-based eye state classification for drowsiness detection."""

from scipy.spatial import distance as dist

import config


class EyeDetector:
    def __init__(self):
        self.counter = 0

    def _compute_eye_score(self, eye_points):
        """Compute eye openness score from 6 eye landmark points."""
        vertical_1 = dist.euclidean(eye_points[1], eye_points[5])
        vertical_2 = dist.euclidean(eye_points[2], eye_points[4])
        horizontal = dist.euclidean(eye_points[0], eye_points[3])
        eye_score = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return eye_score

    def detect(self, landmarks):
        """Detect drowsiness from face landmarks.

        Returns:
            (eye_score, is_drowsy)
        """
        left_eye = landmarks[config.LEFT_EYE]
        right_eye = landmarks[config.RIGHT_EYE]

        left_score = self._compute_eye_score(left_eye)
        right_score = self._compute_eye_score(right_eye)
        avg_score = (left_score + right_score) / 2.0

        if avg_score < config.EYE_SCORE_THRESHOLD:
            self.counter += 1
        else:
            self.counter = 0

        is_drowsy = self.counter >= config.EYE_SCORE_CONSECUTIVE_FRAMES
        return avg_score, is_drowsy
