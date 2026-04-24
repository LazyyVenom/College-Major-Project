"""CNN-based mouth state classification for yawn detection."""

from scipy.spatial import distance as dist

import config


class YawnDetector:
    def __init__(self):
        self.counter = 0

    def detect(self, landmarks):
        """Detect yawning from face landmarks.

        Returns:
            (yawn_score, is_yawning)
        """
        top = landmarks[config.MOUTH_TOP]
        bottom = landmarks[config.MOUTH_BOTTOM]
        left = landmarks[config.MOUTH_LEFT]
        right = landmarks[config.MOUTH_RIGHT]

        vertical = dist.euclidean(top, bottom)
        horizontal = dist.euclidean(left, right)
        yawn_score = vertical / horizontal

        if yawn_score > config.YAWN_THRESHOLD:
            self.counter += 1
        else:
            self.counter = 0

        is_yawning = self.counter >= config.YAWN_CONSECUTIVE_FRAMES
        return yawn_score, is_yawning
