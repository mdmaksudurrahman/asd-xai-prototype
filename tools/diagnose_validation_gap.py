#!/usr/bin/env python3
"""
Diagnose why the web app's test-set results differ from the thesis
(follow-up to tools/run_lab_validation.py).

This is a DIAGNOSTIC: it looks for what reproduces the thesis confusion
matrix, so the cause can be understood and explained. It does not change
the app, and a variant that happens to match is a clue to investigate —
not a setting to adopt just because it gives the published number.

It does four things on the labelled test images:

  1. IMAGE SIZES      How big are the source images? If they are all
                      already the model's input size, resizing cannot be
                      the explanation.
  2. PREPROCESSING    Re-scores every image under several resize methods
     VARIANTS         (the model, labels and threshold unchanged) and
                      reports which, if any, reproduce the thesis
                      confusion matrix exactly.
  3. THRESHOLD SWEEP  Using the app's current scores, which decision
                      thresholds (if any) reproduce the thesis matrix?
  4. WORST ERRORS     The most confidently wrong images. Borderline errors
                      point to a small preprocessing shift; confident
                      errors point to a different model or different data.

USAGE
-----
    python tools/diagnose_validation_gap.py
    python tools/diagnose_validation_gap.py --limit 40      # quick check

Writes reports/diagnose_scores.csv (filenames and scores only, no images).
Takes a few minutes on CPU for the full set.
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402
import lab_validation  # noqa: E402

THESIS_CONFUSION = lab_validation.THESIS_REFERENCE["confusion_matrix"]
THESIS_AUC_PCT = round(lab_validation.THESIS_REFERENCE["auc"] * 100, 2)

# The first entry is what the app (and the thesis XAI notebook) uses today.
PIL_VARIANTS = {
    "pil_nearest": Image.NEAREST,
    "pil_bilinear": Image.BILINEAR,
    "pil_bicubic": Image.BICUBIC,
    "pil_lanczos": Image.LANCZOS,
    "pil_box": Image.BOX,
    "pil_hamming": Image.HAMMING,
}
TF_VARIANT = "tf_bilinear"  # what image_dataset_from_directory does by default
VARIANT_ORDER = list(PIL_VARIANTS) + [TF_VARIANT]
BASELINE = "pil_nearest"


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------
def confusion_at(y_true, p_autistic, threshold=0.5):
    """Confusion matrix [[TN, FP], [FN, TP]] when 'Autistic' means
    p_autistic > threshold (strictly — the same rule the app uses, since
    argmax on [1-p, p] picks class 1 only when p > 0.5)."""
    y_pred = (np.asarray(p_autistic) > threshold).astype(int)
    return confusion_matrix(np.asarray(y_true), y_pred, labels=[0, 1]).tolist()


def matching_thresholds(y_true, p_autistic, target=None, grid=None):
    """Thresholds on the grid whose confusion matrix equals `target`."""
    target = target or THESIS_CONFUSION
    if grid is None:
        grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    return [float(t) for t in grid if confusion_at(y_true, p_autistic, t) == target]


def most_confidently_wrong(names, y_true, p_autistic, threshold=0.5, top=15):
    """Misclassified images, furthest from the threshold first."""
    p = np.asarray(p_autistic)
    wrong = []
    for name, truth, score in zip(names, y_true, p):
        pred = int(score > threshold)
        if pred != truth:
            wrong.append(
                {
                    "filename": name,
                    "true": "Autistic" if truth == 1 else "Non_Autistic",
                    "pred": "Autistic" if pred == 1 else "Non_Autistic",
                    "p_autistic": float(score),
                    "distance": float(abs(score - threshold)),
                }
            )
    wrong.sort(key=lambda r: r["distance"], reverse=True)
    return wrong[:top]


def size_summary(paths, model_size):
    """(most common sizes, fraction already at the model's input size)."""
    sizes = Counter()
    for p in paths:
        with Image.open(p) as im:
            sizes[im.size] += 1
    total = sum(sizes.values())
    already = sizes.get(tuple(model_size), 0)
    return sizes.most_common(5), (already / total if total else 0.0), len(sizes)


# --------------------------------------------------------------------------
# Preprocessing variants and scoring (I/O)
# --------------------------------------------------------------------------
def preprocess_variant(path, variant, width, height):
    """Float32 array in [0, 1], shape (height, width, 3)."""
    with Image.open(path) as im:
        pil = im.convert("RGB")
        if variant in PIL_VARIANTS:
            resized = pil.resize((width, height), resample=PIL_VARIANTS[variant])
            return np.asarray(resized).astype("float32") / 255.0
        if variant == TF_VARIANT:
            import tensorflow as tf

            arr = np.asarray(pil).astype("float32")
            out = tf.image.resize(arr, (height, width), method="bilinear", antialias=False).numpy()
            return out.astype("float32") / 255.0
    raise ValueError(f"unknown variant {variant!r}")


def score_variant(paths, variant, width, height, batch_size=32):
    """p(Autistic) for each path under one preprocessing variant, using
    the app's loaded model and its own output convention."""
    scores = []
    for i in range(0, len(paths), batch_size):
        batch = np.stack([preprocess_variant(p, variant, width, height) for p in paths[i : i + batch_size]])
        raw = app._model.predict(batch.astype("float32"), verbose=0)
        scores.extend((raw[:, 0] if app._binary_output else raw[:, 1]).tolist())
    return np.asarray(scores)


def _list_images(images_dir):
    return sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in app.ALLOWED_EXT
    )


def run_diagnostics(images_dir, out_dir, limit=None, log=print):
    if not app.model_ready():
        raise RuntimeError("Model is not loaded — call app.load_asd_model() first.")

    paths, y_true = [], []
    for p in _list_images(images_dir):
        label = app.infer_true_label_from_filename(p)
        if label is not None:
            paths.append(p)
            y_true.append(1 if label == "Autistic" else 0)
    if limit:
        paths, y_true = paths[:limit], y_true[:limit]
    if not paths:
        raise RuntimeError(f"No labelled images found in {images_dir}")

    names = [os.path.basename(p) for p in paths]
    width, height = app._img_w, app._img_h

    common, frac_native, n_distinct = size_summary(paths, (width, height))

    variants = {}
    for i, name in enumerate(VARIANT_ORDER, 1):
        log(f"  scoring variant {i}/{len(VARIANT_ORDER)}: {name} ...")
        scores = score_variant(paths, name, width, height)
        cm = confusion_at(y_true, scores)
        n = len(y_true)
        auc = float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else None
        variants[name] = {
            "scores": scores,
            "confusion": cm,
            "accuracy": (cm[0][0] + cm[1][1]) / n,
            "auc": auc,
            "exact_match": cm == THESIS_CONFUSION,
            "auc_rounds_to_thesis": auc is not None and round(auc * 100, 2) == THESIS_AUC_PCT,
        }

    baseline_scores = variants[BASELINE]["scores"]
    result = {
        "n": len(paths),
        "sizes": {"common": common, "fraction_native": frac_native, "n_distinct": n_distinct},
        "variants": variants,
        "matching_thresholds": matching_thresholds(y_true, baseline_scores),
        "worst_errors": most_confidently_wrong(names, y_true, baseline_scores),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "diagnose_scores.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label"] + VARIANT_ORDER)
        for idx, name in enumerate(names):
            writer.writerow(
                [name, "Autistic" if y_true[idx] == 1 else "Non_Autistic"]
                + [f"{variants[v]['scores'][idx]:.6f}" for v in VARIANT_ORDER]
            )
    return result


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def print_report(result, log=print):
    n = result["n"]
    s = result["sizes"]
    log("\n" + "=" * 78)
    log(f"1) IMAGE SIZES  ({n} images)")
    log("-" * 78)
    log(
        f"   {s['n_distinct']} distinct sizes; {s['fraction_native'] * 100:.0f}% already at the model's input size"
    )
    for (w, h), count in s["common"]:
        log(f"   {w}x{h}: {count} images")

    log("\n" + "=" * 78)
    log("2) PREPROCESSING VARIANTS  (model, labels and 0.5 threshold unchanged)")
    log("-" * 78)
    log(f"   {'variant':<14}{'accuracy':>10}{'AUC':>9}   {'confusion':<22}vs thesis")
    for name in VARIANT_ORDER:
        v = result["variants"][name]
        auc = "n/a" if v["auc"] is None else f"{v['auc'] * 100:.2f}%"
        flag = "EXACT MATCH" if v["exact_match"] else ""
        if v["auc_rounds_to_thesis"]:
            flag = (flag + "  " if flag else "") + "AUC rounds to thesis"
        mark = "  <- current app" if name == BASELINE else ""
        log(f"   {name:<14}{v['accuracy'] * 100:>9.2f}%{auc:>9}   {str(v['confusion']):<22}{flag}{mark}")
    log(f"   {'thesis':<14}{97.5:>9.2f}%{THESIS_AUC_PCT:>8.2f}%   {str(THESIS_CONFUSION):<22}")

    log("\n" + "=" * 78)
    log("3) THRESHOLD SWEEP  (current app scores, thresholds 0.01-0.99)")
    log("-" * 78)
    thr = result["matching_thresholds"]
    if thr:
        log(f"   Thresholds reproducing the thesis matrix: {thr[0]:.2f} to {thr[-1]:.2f} ({len(thr)} values)")
    else:
        log("   No threshold reproduces the thesis confusion matrix.")

    log("\n" + "=" * 78)
    log("4) MOST CONFIDENTLY WRONG  (current app, threshold 0.5)")
    log("-" * 78)
    if result["worst_errors"]:
        log(f"   {'file':<28}{'true':<14}{'predicted':<14}p(Autistic)")
        for r in result["worst_errors"]:
            log(f"   {r['filename']:<28}{r['true']:<14}{r['pred']:<14}{r['p_autistic']:.3f}")
    else:
        log("   No errors.")

    log("\n" + "=" * 78)
    log("WHAT THIS SUGGESTS")
    log("-" * 78)
    for line in interpret(result):
        log("   " + line)
    log("=" * 78)


def interpret(result):
    """Plain-language reading of the diagnostics. Deliberately hedged:
    these are pointers for investigation, not verdicts."""
    lines = []
    exact = [n for n in VARIANT_ORDER if result["variants"][n]["exact_match"]]
    native = result["sizes"]["fraction_native"]

    if native >= 0.99:
        lines.append("Nearly every image is already at the model's input size, so resize method")
        lines.append("cannot explain the gap; look at the model file or the image files instead.")
    if exact and BASELINE not in exact:
        lines.append(f"Reproduces the thesis matrix exactly: {', '.join(exact)}.")
        lines.append("That points to the thesis accuracy table having used a different resize")
        lines.append("than the XAI notebook (which the app matches). Worth raising with Sefat:")
        lines.append("the accuracy table and the XAI figures may have used different pipelines.")
    elif exact:
        lines.append("The app's current preprocessing already reproduces the thesis matrix.")
    elif native < 0.99:
        lines.append("No resize variant reproduces the thesis matrix exactly.")
    if result["matching_thresholds"] and not exact:
        lines.append("Some decision thresholds reproduce the matrix, but AUC differs from the")
        lines.append("thesis too, so a threshold alone does not explain the gap.")
    if not exact and not result["matching_thresholds"]:
        lines.append("Neither a resize variant nor a threshold reproduces the thesis matrix.")
        lines.append("Remaining explanations: a different model checkpoint than the one used for")
        lines.append("the thesis table (e.g. last epoch vs best), or different image files.")
    errors = result["worst_errors"]
    if errors:
        borderline = sum(1 for e in errors if e["distance"] < 0.2)
        lines.append(f"{borderline} of the {len(errors)} listed errors are borderline (within 0.2 of the")
        lines.append("threshold); confident errors point more to a model or data difference.")
    lines.append("Do not adopt a variant or threshold just because it gives the published number.")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=app.MODEL_PATH, help="Path to the trained model (.h5)")
    parser.add_argument("--images-dir", default=app.TEST_DIR, help="Folder of labelled test images")
    parser.add_argument("--out-dir", default=app.REPORTS_DIR, help="Where to write diagnose_scores.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N images")
    args = parser.parse_args()

    app.MODEL_PATH = args.model
    print(f"Loading model from {args.model} ...")
    if not app.load_asd_model():
        print(f"ERROR: could not load model at {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.images_dir):
        print(f"ERROR: images folder not found: {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Model loaded ({app._img_w}x{app._img_h}). Scoring images from {args.images_dir} ...")
    try:
        result = run_diagnostics(args.images_dir, args.out_dir, limit=args.limit)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print_report(result)
    print(f"\nPer-image scores for every variant: {os.path.join(args.out_dir, 'diagnose_scores.csv')}")


if __name__ == "__main__":
    main()
