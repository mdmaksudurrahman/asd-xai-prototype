#!/usr/bin/env python3
"""
Calibrate TARGET_FACE_FRACTION (face_detection.py) against your real
280 test images.

Part B / Task 2 of the build instructions: "To choose the margin, run
the detector on the 280 test images. Measure the average face-box size
relative to the image, and pick the margin that makes upload crops look
the same." This script does exactly that measurement.

It does NOT modify your test images — it only reads them and reports
statistics.

USAGE
-----
    python tools/calibrate_face_crop_margin.py --images-dir data/test

OUTPUT
------
Printed mean/median face-width-fraction and face-height-fraction across
all images where a face was found, plus a recommended
TARGET_FACE_FRACTION value to paste into face_detection.py. Also saves a
per-image CSV (tools/calibration_output.csv) so you can sanity-check
outliers.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402
import face_detection as fd  # noqa: E402

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images-dir", default="data/test", help="Folder of real images to measure")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_output.csv"),
        help="Where to save the per-image measurements",
    )
    args = parser.parse_args()

    if not fd.ensure_yunet_loaded():
        print(f"ERROR: could not load YuNet ({fd.yunet_status()})", file=sys.stderr)
        print("Run tools/verify_yunet_model.py first to diagnose.", file=sys.stderr)
        sys.exit(1)

    files = sorted(f for f in os.listdir(args.images_dir) if os.path.splitext(f)[1].lower() in ALLOWED_EXT)
    if not files:
        print(f"ERROR: no images found in {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Measuring face size in {len(files)} images from {args.images_dir} ...")

    width_fractions = []
    height_fractions = []
    rows = []
    n_no_face, n_multi_face = 0, 0

    for i, fname in enumerate(files, 1):
        path = os.path.join(args.images_dir, fname)
        try:
            _, arr01 = app.load_preprocess(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(files)}] {fname}: could not read ({exc}), skipping")
            continue

        # Use the image at its OWN native size for measurement, not resized
        # to the model's input — re-load without forcing model input size.
        from PIL import Image

        pil = Image.open(path).convert("RGB")
        import numpy as np

        rgb01_native = np.asarray(pil).astype("float32") / 255.0
        img_h, img_w = rgb01_native.shape[:2]

        detections, used_fallback = fd.detect_faces(rgb01_native)

        if len(detections) == 0:
            n_no_face += 1
            rows.append({"filename": fname, "status": "no_face"})
            print(f"  [{i}/{len(files)}] {fname}: no face detected")
            continue
        if len(detections) > 1:
            n_multi_face += 1
            rows.append({"filename": fname, "status": "multiple_faces", "n_faces": len(detections)})
            print(f"  [{i}/{len(files)}] {fname}: {len(detections)} faces detected, skipping")
            continue

        fx, fy, fw, fh = detections[0].bbox
        wf, hf = fw / img_w, fh / img_h
        width_fractions.append(wf)
        height_fractions.append(hf)
        rows.append(
            {
                "filename": fname,
                "status": "ok",
                "image_w": img_w,
                "image_h": img_h,
                "face_w": fw,
                "face_h": fh,
                "width_fraction": round(wf, 4),
                "height_fraction": round(hf, 4),
                "used_fallback": used_fallback,
            }
        )

    if not width_fractions:
        print("\nERROR: no usable face measurements across any image.", file=sys.stderr)
        sys.exit(1)

    import statistics

    wf_mean, wf_median = statistics.mean(width_fractions), statistics.median(width_fractions)
    hf_mean, hf_median = statistics.mean(height_fractions), statistics.median(height_fractions)
    recommended = round((wf_median + hf_median) / 2, 2)

    print("\n" + "=" * 60)
    print(f"Images measured successfully: {len(width_fractions)} / {len(files)}")
    print(f"No face detected:   {n_no_face}")
    print(f"Multiple faces:      {n_multi_face}")
    print("-" * 60)
    print(f"Face width fraction:  mean={wf_mean:.3f}  median={wf_median:.3f}")
    print(f"Face height fraction: mean={hf_mean:.3f}  median={hf_median:.3f}")
    print("=" * 60)
    print(f"\nRecommended TARGET_FACE_FRACTION = {recommended}")
    print("Paste this into face_detection.py, replacing the current placeholder (0.65):")
    print(f"\n    TARGET_FACE_FRACTION = {recommended}\n")

    import pandas as pd

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Per-image measurements saved to {args.out}")

    if n_no_face > len(files) * 0.1:
        print(
            f"\nNOTE: no face was found in {n_no_face}/{len(files)} images (>10%). "
            "Worth spot-checking a few of those — either they're genuinely hard images, "
            "or something about this dataset's images needs a lower score threshold.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
