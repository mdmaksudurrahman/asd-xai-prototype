#!/usr/bin/env python3
"""
Verify that the YuNet face-detection model is correctly downloaded and
loadable, before relying on it elsewhere.

Run this right after downloading face_detection_yunet_2023mar.onnx into
models/ — it catches the most common failure mode (GitHub serving a tiny
Git LFS *pointer* file instead of the real ~230 KB binary) with a clear
message, rather than a cryptic OpenCV ONNX-parsing error showing up later.

USAGE
-----
    python tools/verify_yunet_model.py
    python tools/verify_yunet_model.py --model path/to/other.onnx
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import face_detection as fd  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=fd.YUNET_MODEL_PATH, help="Path to the YuNet .onnx file")
    args = parser.parse_args()

    print(f"Checking {args.model} ...")

    if not os.path.exists(args.model):
        print(f"FAIL: no file found at {args.model}")
        print("Download face_detection_yunet_2023mar.onnx into models/ first — see README.")
        sys.exit(1)

    size = os.path.getsize(args.model)
    print(f"File size: {size:,} bytes")

    problem = fd._verify_yunet_file(args.model)
    if problem:
        print(f"FAIL: {problem}")
        sys.exit(1)

    print("File size looks right (not an LFS pointer). Trying to load it with OpenCV ...")

    import cv2
    import numpy as np

    try:
        detector = cv2.FaceDetectorYN.create(
            args.model, "", (320, 320), fd.SCORE_THRESHOLD, fd.NMS_THRESHOLD, fd.TOP_K
        )
    except cv2.error as exc:
        print(f"FAIL: OpenCV could not load the model: {exc}")
        sys.exit(1)

    print("Model loaded. Running a smoke-test detection on a blank image ...")
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    try:
        detector.setInputSize((320, 320))
        _, faces = detector.detect(blank)
    except cv2.error as exc:
        print(f"FAIL: model loaded but detect() raised an error: {exc}")
        sys.exit(1)

    n_faces = 0 if faces is None else len(faces)
    print(
        f"detect() ran successfully (found {n_faces} face(s) in a blank image — 0 is expected and correct)."
    )
    print("\nPASS: YuNet is correctly installed and working.")


if __name__ == "__main__":
    main()
