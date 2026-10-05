#!/usr/bin/env python3
"""
Reproduce the thesis's Tables 4.7-4.9 for the 10 reference samples and
check each deterministic metric against the published values.

This is Task 1 / Part A step 3 from the TRL 4 build instructions. Run it
with the REAL model and the 10 thesis sample images (by filename — the
originals were on Google Drive; this script only needs the same
filenames, found in whatever --images-dir you point it at).

USAGE
-----
    python tools/reproduce_thesis_tables.py \\
        --model models/Xception_best.h5 \\
        --images-dir /path/to/folder/containing/the/10/images

The 10 required filenames (from tools/reference/XAI_10_metadata.csv) are:
    Autistic (10).jpg, Autistic (138).jpg, Non-Autistic (7).jpg,
    Non_Autistic.26.jpg, Autistic (52).jpg, Autistic (15).jpg,
    Non-Autistic (63).jpg, Autistic (125).jpg, Autistic (40).jpg,
    Autistic.28.jpg
(4 of these came from the thesis's "valid" split rather than "test" —
if you don't have them to hand, copy them out of wherever your original
Kaggle/training data lives; they don't need to be in data/test/.)

WHAT "MATCH" MEANS
-------------------
Every metric that is deterministic given the model and the image
(prediction, confidence, Grad-CAM/Score-CAM Deletion/Insertion AUC,
Grad-CAM vs Score-CAM IoU/Spearman) should match the published value
within --tol (default 0.01), per the build instructions.

LIME-involving metrics (Spearman/IoU vs LIME) are NOT compared — LIME is
stochastic (re-running it gives slightly different superpixel weights
even with the same num_samples), so the instructions explicitly say to
treat them as reference-only. They're still printed, just without a
MATCH/MISMATCH verdict.

OUTPUT
------
A per-sample, per-metric table printed to stdout, and the same data
written to tools/reproduce_thesis_tables_output.csv for Sefat.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")

# Maps a column in faithfulness_agreement_metrics.csv -> how to get the
# equivalent value out of app.explain_image()'s "full" mode result.
DETERMINISTIC_METRICS = {
    "GradCAM_DeletionAUC": lambda r: r["faithfulness"]["GradCAM_DeletionAUC"],
    "GradCAM_InsertionAUC": lambda r: r["faithfulness"]["GradCAM_InsertionAUC"],
    "ScoreCAM_DeletionAUC": lambda r: r["faithfulness"]["ScoreCAM_DeletionAUC"],
    "ScoreCAM_InsertionAUC": lambda r: r["faithfulness"]["ScoreCAM_InsertionAUC"],
    "Spearman_Grad_vs_Score": lambda r: r["overlap"]["grad_vs_score"]["spearman"],
    "Top10Overlap_Grad_vs_Score": lambda r: r["overlap"]["grad_vs_score"]["iou"],
}

# Reference-only (LIME is stochastic) — printed, never MATCH/MISMATCH.
LIME_METRICS = {
    "Spearman_Grad_vs_LIME": lambda r: r["overlap"]["grad_vs_lime"]["spearman"],
    "Spearman_Score_vs_LIME": lambda r: r["overlap"]["score_vs_lime"]["spearman"],
    "Top10Overlap_Grad_vs_LIME": lambda r: r["overlap"]["grad_vs_lime"]["iou"],
    "Top10Overlap_Score_vs_LIME": lambda r: r["overlap"]["score_vs_lime"]["iou"],
}


def find_image(images_dir, reference_path):
    """The reference CSV stores the original Google Drive path; only the
    filename is portable, so look it up by basename in --images-dir."""
    basename = os.path.basename(reference_path)
    candidate = os.path.join(images_dir, basename)
    if os.path.exists(candidate):
        return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=app.MODEL_PATH, help="Path to the real trained model")
    parser.add_argument(
        "--images-dir", required=True, help="Folder containing the 10 reference images (by filename)"
    )
    parser.add_argument("--tol", type=float, default=0.01, help="Absolute tolerance for MATCH (default 0.01)")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "reproduce_thesis_tables_output.csv"
        ),
        help="Where to save the per-sample results CSV",
    )
    args = parser.parse_args()

    app.MODEL_PATH = args.model
    print(f"Loading model from {args.model} ...")
    if not app.load_asd_model():
        print(f"ERROR: could not load model at {args.model}", file=sys.stderr)
        sys.exit(1)
    print(f"Model loaded. Input size: {app._img_w}x{app._img_h} | last conv layer: {app._last_conv_name}")
    if app._last_conv_name and not app._last_conv_name.endswith("_act"):
        print(
            f"WARNING: expected an activation layer (e.g. '..._sepconv2_act') but got "
            f"'{app._last_conv_name}'. If this isn't an Xception-family model this may be "
            "fine, otherwise double check A1 hasn't regressed.",
            file=sys.stderr,
        )

    meta = pd.read_csv(os.path.join(REFERENCE_DIR, "XAI_10_metadata.csv"))
    faith = pd.read_csv(os.path.join(REFERENCE_DIR, "faithfulness_agreement_metrics.csv"))
    ref = meta.merge(faith, on=["idx", "path", "true_label", "pred_label", "conf"], how="inner")

    all_rows = []
    n_match, n_mismatch, n_missing = 0, 0, 0

    for _, row in ref.iterrows():
        idx = int(row["idx"])
        img_path = find_image(args.images_dir, row["path"])
        print(f"\n--- Sample {idx}: {os.path.basename(row['path'])} ---")

        if img_path is None:
            print(f"  MISSING — not found in {args.images_dir}")
            n_missing += 1
            all_rows.append({"idx": idx, "filename": os.path.basename(row["path"]), "status": "missing"})
            continue

        with open(img_path, "rb") as f:
            result = app.explain_image(f.read(), mode="full", true_label_display=row["true_label"])

        # Prediction + confidence (deterministic)
        pred_matches = result["prediction"].replace("-", "_") == row["pred_label"].replace("-", "_")
        conf_computed = result["confidence"] / 100.0
        conf_ref = float(row["conf"])
        conf_matches = abs(conf_computed - conf_ref) <= args.tol
        print(
            f"  prediction: {'MATCH' if pred_matches else 'MISMATCH'} "
            f"(got {result['prediction']}, expected {row['pred_label']})"
        )
        print(
            f"  confidence: {'MATCH' if conf_matches else 'MISMATCH'} "
            f"(got {conf_computed:.4f}, expected {conf_ref:.4f}, tol={args.tol})"
        )
        n_match += int(pred_matches) + int(conf_matches)
        n_mismatch += int(not pred_matches) + int(not conf_matches)

        out_row = {
            "idx": idx,
            "filename": os.path.basename(row["path"]),
            "status": "ok",
            "prediction_match": pred_matches,
            "confidence_match": conf_matches,
        }

        for col, getter in DETERMINISTIC_METRICS.items():
            computed = getter(result)
            expected = float(row[col])
            matches = abs(computed - expected) <= args.tol
            n_match += int(matches)
            n_mismatch += int(not matches)
            print(
                f"  {col}: {'MATCH' if matches else 'MISMATCH'} "
                f"(got {computed:.4f}, expected {expected:.4f})"
            )
            out_row[col] = computed
            out_row[f"{col}_expected"] = expected
            out_row[f"{col}_match"] = matches

        for col, getter in LIME_METRICS.items():
            computed = getter(result)
            expected = float(row[col])
            print(f"  {col}: REFERENCE ONLY (got {computed:.4f}, thesis run was {expected:.4f})")
            out_row[col] = computed
            out_row[f"{col}_expected"] = expected

        all_rows.append(out_row)

    print("\n" + "=" * 60)
    print(f"Deterministic metrics: {n_match} MATCH, {n_mismatch} MISMATCH, {n_missing} images missing")
    print("=" * 60)

    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"\nFull results saved to {args.out}")

    if n_mismatch > 0 or n_missing > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
