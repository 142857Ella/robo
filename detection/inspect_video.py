"""Print basic metadata for a video used by the detection assignment."""

import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description="Inspect a video file")
    parser.add_argument("video", help="Path to an input video, e.g. assets/small.mp4")
    args = parser.parse_args()

    # VideoCapture opens the video decoder. It returns no usable frames if the
    # path is wrong or the video codec cannot be read.
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    # These properties come from the video container and help us verify that
    # the supplied clips match the assignment description.
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Actually decode one frame: metadata alone does not prove frame reading
    # works. OpenCV stores each decoded frame as a BGR image array.
    success, frame = capture.read()
    capture.release()

    if not success:
        raise SystemExit(f"Could not read the first frame: {args.video}")

    print(f"Video: {args.video}")
    print(f"Size: {width} x {height}")
    print(f"FPS: {fps:g}")
    print(f"Frames: {frame_count}")
    print(f"First frame array: {frame.shape}, dtype={frame.dtype}")


if __name__ == "__main__":
    main()
