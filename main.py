"""Driver Safety System - Main Entry Point."""

import cv2
import time

import config
from utils.landmarks import FaceMeshDetector
from detectors.eye_detector import EyeDetector
from detectors.yawn_detector import YawnDetector
from detectors.head_pose_detector import HeadPoseDetector
from detectors.phone_detector import PhoneDetector
from detectors.seatbelt_detector import SeatbeltDetector
from detectors.drunk_detector import DrunkDetector
from alerts.alert_manager import AlertManager
from utils.drawing import draw_hud


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    face_mesh = FaceMeshDetector()
    eye_det = EyeDetector()
    yawn_det = YawnDetector()
    head_det = HeadPoseDetector()
    phone_det = PhoneDetector()
    seatbelt_det = SeatbeltDetector() if config.SEATBELT_ENABLED else None
    drunk_det = DrunkDetector()
    alert_mgr = AlertManager()

    phone_det.start()
    if seatbelt_det:
        seatbelt_det.start()

    prev_time = time.time()
    fps = 0

    print("Driver Safety System started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # MediaPipe landmarks (single call, shared by detectors)
        landmarks = face_mesh.get_landmarks(rgb)

        active_alerts = []
        eye_score, yawn_score, yaw, pitch, drunk_score = 0.0, 0.0, 0.0, 0.0, 0.0

        if landmarks is not None:
            eye_score, is_drowsy = eye_det.detect(landmarks)
            yawn_score, is_yawning = yawn_det.detect(landmarks)
            yaw, pitch, _, is_distracted = head_det.detect(landmarks, frame.shape)
            drunk_score, is_drunk = drunk_det.detect(landmarks, eye_score, yaw, pitch)

            if is_drowsy:
                active_alerts.append("drowsiness")
            if is_yawning:
                active_alerts.append("yawning")
            if is_distracted:
                active_alerts.append("distraction")
            if is_drunk:
                active_alerts.append("drunk")

        # YOLO detectors (threaded)
        phone_det.update_frame(frame)
        if phone_det.is_phone_detected:
            active_alerts.append("phone")

        if seatbelt_det:
            seatbelt_det.update_frame(frame)
            if not seatbelt_det.is_wearing:
                active_alerts.append("seatbelt")

        # Fire alerts
        for name in active_alerts:
            alert_mgr.trigger(name)

        # Draw HUD and alerts
        draw_hud(frame, yaw, pitch, fps, active_alerts, drunk_score)
        alert_mgr.draw_warnings(frame, active_alerts)

        cv2.imshow("Driver Safety System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    phone_det.stop()
    if seatbelt_det:
        seatbelt_det.stop()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
