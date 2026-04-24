"""Drunk/impairment detection using multi-signal facial analysis.

Combines several behavioral cues over a sliding window:
1. Eye blink irregularity (eye score variance)
2. Facial asymmetry (left vs right side landmark distances)
3. Head sway (low-frequency oscillation in yaw/pitch)
4. Frequent prolonged eye closures
"""

import collections
import numpy as np
from scipy.spatial import distance as dist

import config


class DrunkDetector:
    def __init__(self):
        window = config.DRUNK_WINDOW_FRAMES
        self.eye_score_history = collections.deque(maxlen=window)
        self.yaw_history = collections.deque(maxlen=window)
        self.pitch_history = collections.deque(maxlen=window)
        self.asymmetry_history = collections.deque(maxlen=window)
        self.blink_durations = collections.deque(maxlen=20)

        self._eye_closed_frames = 0
        self._blink_count = 0
        self._was_closed = False
        self.counter = 0

    def _facial_asymmetry(self, landmarks):
        """Measure asymmetry between left and right face landmarks.

        Higher values indicate more facial drooping/asymmetry.
        """
        # Left vs right eye openness
        left_eye = landmarks[config.LEFT_EYE]
        right_eye = landmarks[config.RIGHT_EYE]

        left_v1 = dist.euclidean(left_eye[1], left_eye[5])
        left_v2 = dist.euclidean(left_eye[2], left_eye[4])
        left_h = dist.euclidean(left_eye[0], left_eye[3])
        left_score = (left_v1 + left_v2) / (2.0 * left_h)

        right_v1 = dist.euclidean(right_eye[1], right_eye[5])
        right_v2 = dist.euclidean(right_eye[2], right_eye[4])
        right_h = dist.euclidean(right_eye[0], right_eye[3])
        right_score = (right_v1 + right_v2) / (2.0 * right_h)

        eye_asymmetry = abs(left_score - right_score)

        # Left vs right mouth corner height difference
        left_mouth = landmarks[config.DRUNK_LEFT_MOUTH]
        right_mouth = landmarks[config.DRUNK_RIGHT_MOUTH]
        nose_tip = landmarks[config.DRUNK_NOSE_TIP]

        left_dist = dist.euclidean(left_mouth, nose_tip)
        right_dist = dist.euclidean(right_mouth, nose_tip)
        mouth_asymmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)

        return (eye_asymmetry + mouth_asymmetry) / 2.0

    def _track_blinks(self, eye_score):
        """Track blink patterns - impaired drivers show irregular blinks."""
        is_closed = eye_score < config.EYE_SCORE_THRESHOLD

        if is_closed:
            self._eye_closed_frames += 1
        else:
            if self._was_closed and self._eye_closed_frames > 0:
                self.blink_durations.append(self._eye_closed_frames)
                self._blink_count += 1
            self._eye_closed_frames = 0

        self._was_closed = is_closed

    def detect(self, landmarks, eye_score, yaw, pitch):
        """Detect signs of impairment from facial metrics.

        Args:
            landmarks: MediaPipe face landmarks array.
            eye_score: Current eye state score from EyeDetector.
            yaw: Current head yaw from HeadPoseDetector.
            pitch: Current head pitch from HeadPoseDetector.

        Returns:
            (drunk_score, is_drunk): Score 0-1 and boolean flag.
        """
        # Collect signals
        self.eye_score_history.append(eye_score)
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)

        asymmetry = self._facial_asymmetry(landmarks)
        self.asymmetry_history.append(asymmetry)

        self._track_blinks(eye_score)

        # Need enough data before scoring
        if len(self.eye_score_history) < config.DRUNK_MIN_FRAMES:
            return 0.0, False

        scores = []

        # Signal 1: Eye score variance (unstable eye opening = impairment)
        eye_arr = np.array(self.eye_score_history)
        eye_std = np.std(eye_arr)
        eye_var_score = min(eye_std / config.DRUNK_EYE_STD_THRESHOLD, 1.0)
        scores.append(eye_var_score * config.DRUNK_WEIGHT_EYE_VAR)

        # Signal 2: Head sway (slow oscillation in yaw/pitch)
        yaw_arr = np.array(self.yaw_history)
        pitch_arr = np.array(self.pitch_history)
        yaw_std = np.std(yaw_arr)
        pitch_std = np.std(pitch_arr)
        sway_magnitude = (yaw_std + pitch_std) / 2.0
        sway_score = min(sway_magnitude / config.DRUNK_SWAY_THRESHOLD, 1.0)
        scores.append(sway_score * config.DRUNK_WEIGHT_SWAY)

        # Signal 3: Facial asymmetry
        asym_arr = np.array(self.asymmetry_history)
        avg_asymmetry = np.mean(asym_arr)
        asym_score = min(avg_asymmetry / config.DRUNK_ASYMMETRY_THRESHOLD, 1.0)
        scores.append(asym_score * config.DRUNK_WEIGHT_ASYMMETRY)

        # Signal 4: Blink irregularity (high variance in blink duration)
        if len(self.blink_durations) >= 3:
            blink_std = np.std(list(self.blink_durations))
            blink_score = min(blink_std / config.DRUNK_BLINK_STD_THRESHOLD, 1.0)
        else:
            blink_score = 0.0
        scores.append(blink_score * config.DRUNK_WEIGHT_BLINK)

        # Signal 5: Prolonged eye closure ratio
        closure_ratio = np.sum(eye_arr < config.EYE_SCORE_THRESHOLD) / len(eye_arr)
        closure_score = min(closure_ratio / config.DRUNK_CLOSURE_RATIO_THRESHOLD, 1.0)
        scores.append(closure_score * config.DRUNK_WEIGHT_CLOSURE)

        # Combined weighted score
        total_weight = (config.DRUNK_WEIGHT_EYE_VAR + config.DRUNK_WEIGHT_SWAY +
                        config.DRUNK_WEIGHT_ASYMMETRY + config.DRUNK_WEIGHT_BLINK +
                        config.DRUNK_WEIGHT_CLOSURE)
        drunk_score = sum(scores) / total_weight

        if drunk_score >= config.DRUNK_THRESHOLD:
            self.counter += 1
        else:
            self.counter = 0

        is_drunk = self.counter >= config.DRUNK_CONSECUTIVE_FRAMES
        return drunk_score, is_drunk
