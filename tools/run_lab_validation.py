#!/usr/bin/env python3
"""
Run the full test set through the web app's own code and compare the
result with the numbers published in the thesis (Task 5 / Part D).

Writes three files into --out-dir (default: reports/):
    lab_validation.html   the report (also served at /lab by the web app)
    lab_validation.csv    one row per image
    lab_validation.json   everything, machine-readable

The reports contain filenames and numbers only — never images. Consider
adding reports/ to .gitignore.

USAGE
-----
    python tools/run_lab_validation.py
    python tools/run_lab_validation.py --limit 20      # quick smoke test
    python tools/run_lab_validation.py --model models/Xception_best.h5 --images-dir data/test

A --limit run is a partial run: it is reported as "not comparable" to the
thesis rather than given a MATCH/DIFF verdict.

Exit code: 0 on success; 1 if the run covered all images but any metric
or the confusion matrix differs from the thesis, or on a setup error.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402
import lab_validation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=app.MODEL_PATH, help="Path to the trained model (.h5)")
    parser.add_argument("--images-dir", default=app.TEST_DIR, help="Folder of labelled test images")
    parser.add_argument("--out-dir", default=app.REPORTS_DIR, help="Where to write the reports")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N images (smoke test)")
    args = parser.parse_args()

    app.MODEL_PATH = args.model
    print(f"Loading model from {args.model} ...")
    if not app.load_asd_model():
        print(f"ERROR: could not load model at {args.model}", file=sys.stderr)
        sys.exit(1)
    print(f"Model loaded. Input size {app._img_w}x{app._img_h}, last conv layer {app._last_conv_name}")

    if not os.path.isdir(args.images_dir):
        print(f"ERROR: images folder not found: {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Running images from {args.images_dir} ...")
    try:
        report = lab_validation.run_validation(args.images_dir, args.out_dir, limit=args.limit)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    cmp_ = report["comparison"]
    print("\n" + "=" * 62)
    print(f"{'Metric':<12}{'Web app':>10}{'Thesis':>10}{'Diff (pp)':>12}   Verdict")
    print("-" * 62)
    for r in cmp_["metrics"]:
        ours = "n/a" if r["ours"] is None else f"{r['ours'] * 100:.2f}%"
        diff = "n/a" if r["diff"] is None else f"{r['diff'] * 100:+.2f}"
        verdict = r["status"] if cmp_["comparable"] else "not comparable"
        print(
            f"{lab_validation.METRIC_LABELS[r['metric']]:<12}{ours:>10}{r['thesis'] * 100:>9.2f}%{diff:>12}   {verdict}"
        )
    print("-" * 62)
    print(f"Confusion matrix (web app): {cmp_['confusion']['ours']}")
    print(f"Confusion matrix (thesis):  {cmp_['confusion']['thesis']}")

    att = report["attention"]
    print("-" * 62)
    mean = "n/a" if att["overall"]["mean"] is None else f"{att['overall']['mean']}%"
    print(f"Attention on face: mean {mean} over {att['n_with_face']} images with a face")
    print(
        f"No usable face: {att['n_without_face']} images | mostly off-face: {len(att['off_face_images'])} images"
    )
    if report["skipped"]:
        print(f"Skipped (unreadable or unlabelled): {len(report['skipped'])} images")
    print("=" * 62)
    print(f"\nReports written to {args.out_dir}")

    if not cmp_["comparable"]:
        print(f"NOTE: partial run ({report['metrics']['n']} images) — not comparable to the thesis.")
        return
    if cmp_["all_match"]:
        print("RESULT: all thesis metrics and the confusion matrix are reproduced.")
        return
    print("RESULT: DIFFERENCES from the thesis found — see the table above.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
