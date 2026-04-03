"""MediaPipe FaceMesh initialization and landmark extraction."""

import mediapipe as mp
import numpy as np


class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

    def get_landmarks(self, rgb_frame):
        """Process an RGB frame and return landmark coordinates.

        Returns:
            numpy array of shape (468, 3) with (x, y, z) in pixel coordinates,
            or None if no face detected.
        """
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        h, w = rgb_frame.shape[:2]
        landmarks = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in face.landmark],
            dtype=np.float64,
        )
        return landmarks

    def close(self):
        self.face_mesh.close()
