"""Run the ONNX detector on the first frame of a video."""

import argparse
import json

import cv2
import onnxruntime as ort

from preprocess import letterbox
from decode import decode_predictions
from nms import non_max_suppression
from pose import estimate_pose, load_calibration


def main():
    parser = argparse.ArgumentParser(description="Detect armor plates in a video and write JSONL")
    parser.add_argument("--model", required=True, help="Path to the ONNX model")
    parser.add_argument("--video", required=True, help="Path to an input video")
    parser.add_argument("--calibration", required=True, help="Path to camera calibration YAML")
    parser.add_argument("--output", required=True, help="Path for the output JSONL file")
    args = parser.parse_args()

    camera_matrix, distortion = load_calibration(args.calibration)
    # Use CPU explicitly so the script runs without requiring a GPU runtime.
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # ONNX models name their input and output tensors. Read those names rather
    # than assuming what the exporter chose.
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    print(f"Model input:  {model_input.name}, shape={model_input.shape}, type={model_input.type}")
    print(f"Model output: {model_output.name}, shape={model_output.shape}, type={model_output.type}")

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    frame_index = 0
    with open(args.output, "w", encoding="utf-8") as output_file:
        while True:
            success, frame = capture.read()
            if not success:
                break

            # Preprocess and infer one frame at a time to keep memory use low.
            tensor, scale, pad_x, pad_y = letterbox(frame)
            (prediction,) = session.run([model_output.name], {model_input.name: tensor})
            candidates = decode_predictions(prediction, scale, pad_x, pad_y, threshold=0.5)
            detections = non_max_suppression(candidates, iou_threshold=0.5)

            frame_detections = []
            for detection in detections:
                corners_camera, reprojection_error = estimate_pose(
                    detection, camera_matrix, distortion
                )
                frame_detections.append({
                    "color": detection["color"],
                    "pattern": detection["pattern"],
                    "confidence": detection["confidence"],
                    "corners_px": detection["corners_px"],
                    "corners_camera_m": corners_camera,
                    "reprojection_error_px": reprojection_error,
                })

            # One JSON object per line makes the file streamable and matches
            # the assignment format. Timestamps are based on the stated 30 FPS.
            record = {
                "frame_index": frame_index,
                "timestamp_s": frame_index / 30.0,
                "detections": frame_detections,
            }
            output_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            frame_index += 1

    capture.release()
    print(f"Wrote {frame_index} frames to {args.output}")


if __name__ == "__main__":
    main()
