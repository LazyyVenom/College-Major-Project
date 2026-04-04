"""Centralized alert system with cooldowns, priority, and audio playback."""

import os
import time

import cv2
import numpy as np

import config

# Generate alert sounds programmatically if wav files don't exist
SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")


def _generate_beep(filename, frequency=800, duration_ms=500, sample_rate=44100):
    """Generate a simple beep WAV file if it doesn't exist."""
    filepath = os.path.join(SOUNDS_DIR, filename)
    if os.path.exists(filepath):
        return filepath

    import wave

    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    samples = (32767 * 0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

    os.makedirs(SOUNDS_DIR, exist_ok=True)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return filepath


class AlertManager:
    def __init__(self):
        import pygame
        pygame.mixer.init()
        self._pygame = pygame

        # Generate default alert sounds with different frequencies
        freqs = {
            "drowsy_alert.wav": 1000,
            "yawn_alert.wav": 600,
            "distraction_alert.wav": 900,
            "phone_alert.wav": 750,
            "seatbelt_alert.wav": 500,
        }

        self.sounds = {}
        for name, cfg in config.ALERTS.items():
            sound_file = os.path.basename(cfg["sound"])
            freq = freqs.get(sound_file, 800)
            filepath = _generate_beep(sound_file, frequency=freq)
            self.sounds[name] = pygame.mixer.Sound(filepath)

        self.last_alert_time = {name: 0.0 for name in config.ALERTS}
        self.text_sizes = {
            name: cv2.getTextSize(cfg["message"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            for name, cfg in config.ALERTS.items()
        }

    def trigger(self, alert_name):
        """Trigger an alert if not in cooldown. Returns True if alert fired."""
        now = time.time()
        cfg = config.ALERTS[alert_name]

        if now - self.last_alert_time[alert_name] < cfg["cooldown"]:
            return False

        self.last_alert_time[alert_name] = now

        # Only play sound for highest priority active alert
        if not self._pygame.mixer.get_busy():
            self.sounds[alert_name].play()

        return True

    def draw_warnings(self, frame, active_alerts):
        """Draw visual warnings on the frame for all active alerts."""
        if not active_alerts:
            return

        h, w = frame.shape[:2]

        # Sort by priority (highest first)
        sorted_alerts = sorted(
            active_alerts,
            key=lambda a: config.ALERTS[a]["priority"],
            reverse=True,
        )

        # Draw colored border for highest priority alert
        top_alert = sorted_alerts[0]
        color = config.ALERTS[top_alert]["color"]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)

        # Draw alert messages
        y_offset = 30
        for alert_name in sorted_alerts:
            cfg = config.ALERTS[alert_name]
            msg = cfg["message"]
            color = cfg["color"]

            text_size = self.text_sizes[alert_name]
            cv2.rectangle(
                frame,
                (5, y_offset - 20),
                (text_size[0] + 15, y_offset + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, msg, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            y_offset += 35
