"""Head pose estimation for distraction detection using solvePnP."""

import cv2
import numpy as np

import config


class HeadPoseDetector:
    def __init__(self):
        self.counter = 0
        self.model_points = np.array(config.MODEL_POINTS_3D, dtype=np.float64)

    def detect(self, landmarks, frame_shape):
        """Estimate head pose and detect distraction.

        Returns:
            (yaw, pitch, roll, is_distracted)
        """
        h, w = frame_shape[:2]

        # 2D image points from landmarks
        image_points = np.array(
            [landmarks[i][:2] for i in config.HEAD_POSE_LANDMARKS],
            dtype=np.float64,
        )

        # Camera intrinsics (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            self.counter = 0
            return 0.0, 0.0, 0.0, False

        # Convert rotation vector to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        proj_matrix = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            np.vstack((proj_matrix, [0, 0, 0, 1]))[:3]
        )

        pitch = euler_angles[0, 0]
        yaw = euler_angles[1, 0]
        roll = euler_angles[2, 0]

        if abs(yaw) > config.YAW_THRESHOLD or abs(pitch) > config.PITCH_THRESHOLD:
            self.counter += 1
        else:
            self.counter = 0

        is_distracted = self.counter >= config.HEAD_POSE_CONSECUTIVE_FRAMES
        return yaw, pitch, roll, is_distracted
