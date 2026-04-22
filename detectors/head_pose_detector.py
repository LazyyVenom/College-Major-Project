"""Head pose estimation for distraction detection using solvePnP."""

import cv2
import numpy as np

import config


class HeadPoseDetector:
    def __init__(self):
        self.counter = 0
        self.model_points = np.array(config.MODEL_POINTS_3D, dtype=np.float64)
        self._cached_shape = None
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1))

    def detect(self, landmarks, frame_shape):
        """Estimate head pose and detect distraction.

        Returns:
            (yaw, pitch, roll, is_distracted)
        """
        h, w = frame_shape[:2]

        image_points = np.array(
            [landmarks[i][:2] for i in config.HEAD_POSE_LANDMARKS],
            dtype=np.float64,
        )

        if (h, w) != self._cached_shape:
            focal_length = w
            self._camera_matrix = np.array(
                [[focal_length, 0, w / 2],
                 [0, focal_length, h / 2],
                 [0, 0, 1]],
                dtype=np.float64,
            )
            self._cached_shape = (h, w)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            self.counter = 0
            return 0.0, 0.0, 0.0, False

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        proj_matrix = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = euler_angles[0, 0]
        yaw = euler_angles[1, 0]
        roll = euler_angles[2, 0]

        # Normalize angles — decomposeProjectionMatrix can return values
        # near ±180° instead of near 0° when looking straight ahead.
        if pitch > 90:
            pitch = 180 - pitch
        elif pitch < -90:
            pitch = -180 - pitch

        if yaw > 90:
            yaw = 180 - yaw
        elif yaw < -90:
            yaw = -180 - yaw

        if abs(yaw) > config.YAW_THRESHOLD or abs(pitch) > config.PITCH_THRESHOLD:
            self.counter += 1
        else:
            self.counter = 0

        is_distracted = self.counter >= config.HEAD_POSE_CONSECUTIVE_FRAMES
        return yaw, pitch, roll, is_distracted
