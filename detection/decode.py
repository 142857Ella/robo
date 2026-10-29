"""Decode model rows into labeled armor-plate candidates."""

import numpy as np


COLORS = ["blue", "red", "neutral", "purple"]
PATTERNS = ["sentry", "1", "2", "3", "4", "5", "outpost", "base_small", "base_big"]


def sigmoid(value):
    """Convert a logit to a probability, clipping to avoid exp overflow."""
    value = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-value))


def decode_predictions(prediction, scale, pad_x, pad_y, threshold=0.5):
    """Decode one [25200, 22] output and undo the letterbox transform."""
    rows = prediction[0]  # Remove the batch dimension: [25200, 22].
    scores = sigmoid(rows[:, 8])
    keep = scores >= threshold

    candidates = []
    for row, confidence in zip(rows[keep], scores[keep]):
        # Each row stores four ordered (x, y) pairs in 640x640 model pixels.
        corners = row[0:8].reshape(4, 2).astype(np.float32)

        # Reverse the centered padding first, then reverse the uniform resize.
        corners[:, 0] = (corners[:, 0] - pad_x) / scale
        corners[:, 1] = (corners[:, 1] - pad_y) / scale

        color_id = int(np.argmax(row[9:13]))
        pattern_id = int(np.argmax(row[13:22]))
        candidates.append({
            "color": COLORS[color_id],
            "pattern": PATTERNS[pattern_id],
            "confidence": float(confidence),
            "corners_px": corners.tolist(),
        })

    return candidates
