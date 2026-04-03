"""HUD overlay rendering for the driver safety system."""

import cv2

import config


def draw_hud(frame, ear, mar, yaw, pitch, fps, active_alerts):
    """Draw the status dashboard on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent background panel (right side)
    panel_w = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    x = w - panel_w + 10
    y = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (x, y), font, scale, white, 1)
    y += 25

    # EAR
    ear_color = red if ear < config.EAR_THRESHOLD else green
    cv2.putText(frame, f"EAR: {ear:.2f}", (x, y), font, scale, ear_color, 1)
    y += 25

    # MAR
    mar_color = red if mar > config.YAWN_THRESHOLD else green
    cv2.putText(frame, f"MAR: {mar:.2f}", (x, y), font, scale, mar_color, 1)
    y += 25

    # Head Pose
    yaw_color = red if abs(yaw) > config.YAW_THRESHOLD else green
    cv2.putText(frame, f"Yaw: {yaw:.1f}", (x, y), font, scale, yaw_color, 1)
    y += 25

    pitch_color = red if abs(pitch) > config.PITCH_THRESHOLD else green
    cv2.putText(frame, f"Pitch: {pitch:.1f}", (x, y), font, scale, pitch_color, 1)
    y += 25

    # Status
    if active_alerts:
        status = "UNSAFE"
        status_color = red
    else:
        status = "SAFE"
        status_color = green

    cv2.putText(frame, f"Status: {status}", (x, y), font, 0.6, status_color, 2)
