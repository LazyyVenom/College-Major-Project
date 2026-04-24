"""HUD overlay rendering for the driver safety system."""

import cv2
import numpy as np

import config


def draw_hud(frame, yaw, pitch, fps, active_alerts, drunk_score=0.0):
    """Draw the status dashboard on the frame."""
    h, w = frame.shape[:2]

    panel_w = 220
    panel_h = 155
    panel_x = w - panel_w
    roi = frame[0:panel_h, panel_x:w]
    cv2.addWeighted(np.zeros_like(roi), 0.6, roi, 0.4, 0, roi)

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

    # Head Pose
    yaw_color = red if abs(yaw) > config.YAW_THRESHOLD else green
    cv2.putText(frame, f"Yaw: {yaw:.1f}", (x, y), font, scale, yaw_color, 1)
    y += 25

    pitch_color = red if abs(pitch) > config.PITCH_THRESHOLD else green
    cv2.putText(frame, f"Pitch: {pitch:.1f}", (x, y), font, scale, pitch_color, 1)
    y += 25

    # Drunk Score
    drunk_color = red if drunk_score >= config.DRUNK_THRESHOLD else green
    cv2.putText(frame, f"Drunk: {drunk_score:.2f}", (x, y), font, scale, drunk_color, 1)
    y += 25

    # Status
    if active_alerts:
        status = "UNSAFE"
        status_color = red
    else:
        status = "SAFE"
        status_color = green

    cv2.putText(frame, f"Status: {status}", (x, y), font, 0.6, status_color, 2)
