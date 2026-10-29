"""Draw detections from one JSONL frame back onto its source video frame."""

import argparse
import json

import cv2
import numpy as np


def load_record(jsonl_path, wanted_index):
    """Find and parse the JSON record for the requested frame index."""
    with open(jsonl_path, encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            record = json.loads(line)
            if record["frame_index"] == wanted_index:
                return record
    raise ValueError(f"Frame {wanted_index} is not present in {jsonl_path}")


def draw_record(frame, record):
    """Overlay each detection's ordered corners, class, and confidence."""
    for detection in record["detections"]:
        corners = np.asarray(detection["corners_px"], dtype=np.int32)
        polygon = corners.reshape((-1, 1, 2))
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        for corner_index, (x, y) in enumerate(corners):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(corner_index + 1),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        label = f"{detection['color']} {detection['pattern']} {detection['confidence']:.2f}"
        top_left = tuple(corners[0])
        cv2.putText(
            frame,
            label,
            (int(top_left[0]), int(top_left[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Visualize one JSONL detection frame")
    parser.add_argument("--video", required=True, help="Source video path")
    parser.add_argument("--jsonl", required=True, help="Detection JSONL path")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame to draw (default: 0)")
    parser.add_argument("--output", required=True, help="Path for the annotated PNG")
    args = parser.parse_args()

    record = load_record(args.jsonl, args.frame_index)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise SystemExit(f"Could not read frame {args.frame_index} from {args.video}")

    annotated = draw_record(frame, record)
    if not cv2.imwrite(args.output, annotated):
        raise SystemExit(f"Could not write image: {args.output}")
    print(f"Saved {len(record['detections'])} detections to {args.output}")


if __name__ == "__main__":
    main()
