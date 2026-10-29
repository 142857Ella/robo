"""Resize one BGR video frame into the detector's model input tensor."""

import argparse

import cv2
import numpy as np


INPUT_SIZE = 640
PAD_VALUE = 114


def letterbox(frame_bgr):
    """Return (tensor, scale, pad_x, pad_y) for one decoded video frame."""
    original_height, original_width = frame_bgr.shape[:2]

    # Use one scale for both axes so the image's geometry is not distorted.
    scale = min(INPUT_SIZE / original_width, INPUT_SIZE / original_height)
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    resized = cv2.resize(frame_bgr, (resized_width, resized_height))

    # Center the resized image in a square canvas. Keep the actual integer
    # offsets because they are needed later to undo this transformation.
    pad_x = (INPUT_SIZE - resized_width) // 2
    pad_y = (INPUT_SIZE - resized_height) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    # OpenCV decodes BGR, while this model expects RGB. The model also expects
    # values in [0, 1], float16, and channel-first layout: [1, 3, 640, 640].
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
    tensor = np.ascontiguousarray(tensor, dtype=np.float16)

    return tensor, scale, pad_x, pad_y


def main():
    parser = argparse.ArgumentParser(description="Prepare a video frame for the detector")
    parser.add_argument("video", help="Path to an input video")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    success, frame = capture.read()
    capture.release()
    if not success:
        raise SystemExit(f"Could not read the first frame: {args.video}")

    tensor, scale, pad_x, pad_y = letterbox(frame)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor value range: {tensor.min():.4f} to {tensor.max():.4f}")
    print(f"Scale: {scale:.6f}; padding: left/right={pad_x}, top/bottom={pad_y}")


if __name__ == "__main__":
    main()
