#!/usr/bin/env python3
"""
Compare the web app's probabilities with the ones the thesis XAI
notebook saved from its original Colab run.

The notebook's saved cell outputs contain full-precision p(Autistic) for
about 20 images from the test folder (plus the 10 reference images in
tools/reference/XAI_10_metadata.csv). If the web app reproduces those to
within floating-point noise, then it is running the same model through
the same preprocessing as the notebook — which is what "the web app is
the thesis model" has to mean in practice.

No model or TensorFlow is needed: this only reads a CSV that an earlier
run already wrote (reports/diagnose_scores.csv or
reports/lab_validation.csv).

USAGE
-----
    python tools/compare_with_notebook_outputs.py
    python tools/compare_with_notebook_outputs.py --scores-csv reports/lab_validation.csv
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_CSV = os.path.join(ROOT, "tools", "reference", "XAI_10_metadata.csv")
DEFAULT_SCORE_FILES = [
    os.path.join(ROOT, "reports", "diagnose_scores.csv"),
    os.path.join(ROOT, "reports", "lab_validation.csv"),
]
# diagnose_scores.csv / lab_validation.csv
SCORE_COLUMNS = ("pil_nearest", "p_autistic")

TOL_EQUIVALENT = 1e-4  # floating-point noise (different TF build, CPU vs GPU)
TOL_CLOSE = 1e-2

# p(Autistic) printed in ThesisXAIconsistency.ipynb's saved cell outputs.
# Cell 3's table is printed to 6 decimals; cells 7 and 15 to 8.
NOTEBOOK_P_AUTISTIC = {
    "Autistic (126).jpg": 0.999993,  # cell 3
    "Non-Autistic (54).jpg": 0.000030,  # cell 3
    "Non-Autistic (36).jpg": 0.000050,  # cell 3
    "Non-Autistic (95).jpg": 0.000195,  # cell 3
    "Non-Autistic (85).jpg": 0.000207,  # cell 3
    "Autistic (39).jpg": 0.999735,  # cell 3
    "Autistic (80).jpg": 0.999080,  # cell 3
    "Non-Autistic (51).jpg": 0.001766,  # cell 3
    "Non-Autistic (10).jpg": 0.001880,  # cell 3
    # cell 3 (superseded below by the full-precision CSV value)
    "Autistic (10).jpg": 0.998018,
    # cell 6 — a WRONG prediction in the notebook's own run
    "Non-Autistic (109).jpg": 0.660293,
    "Autistic (1).jpg": 0.9972779,  # cell 7
    # cell 15 — also wrong in the notebook's run
    "Autistic (114).jpg": 0.02096741,
}


def load_xai10_reference(path=REFERENCE_CSV):
    """Full-precision p(Autistic) for the 10 reference images."""
    ref = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref[os.path.basename(row["path"].replace("\\", "/"))] = float(row["p1"])
    return ref


def load_app_scores(path, column=None):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        chosen = column or next((c for c in SCORE_COLUMNS if c in columns), None)
        if chosen is None or chosen not in columns:
            raise ValueError(f"{path} has no score column (looked for {', '.join(SCORE_COLUMNS)}).")
        return {row["filename"]: float(row[chosen]) for row in reader}, chosen


def compare(reference, app_scores):
    """Rows for every reference image the app also scored, plus the
    names that could not be compared (e.g. the reference images that
    came from the validation split, not the test folder)."""
    rows, missing = [], []
    for name, ref_p in sorted(reference.items()):
        if name not in app_scores:
            missing.append(name)
            continue
        app_p = app_scores[name]
        rows.append(
            {
                "filename": name,
                "notebook": ref_p,
                "app": app_p,
                "abs_diff": abs(app_p - ref_p),
                "same_side": (app_p > 0.5) == (ref_p > 0.5),
            }
        )
    return rows, missing


def summarize(rows):
    if not rows:
        return {"n": 0, "max_abs_diff": None, "mean_abs_diff": None, "all_same_side": None, "verdict": "none"}
    diffs = [r["abs_diff"] for r in rows]
    worst = max(diffs)
    verdict = "equivalent" if worst <= TOL_EQUIVALENT else "close" if worst <= TOL_CLOSE else "different"
    return {
        "n": len(rows),
        "max_abs_diff": worst,
        "mean_abs_diff": sum(diffs) / len(diffs),
        "all_same_side": all(r["same_side"] for r in rows),
        "verdict": verdict,
    }


VERDICT_TEXT = {
    "equivalent": "The web app reproduces the notebook's probabilities to within floating-point noise.",
    "close": "Close but not identical — small differences (library versions, hardware?) worth noting.",
    "different": "The web app's probabilities differ from the notebook's — a pipeline difference to investigate.",
    "none": "No reference image was found in the scores file.",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scores-csv", default=None, help="CSV with filename + a score column")
    parser.add_argument("--column", default=None, help="Score column (default: auto-detect)")
    args = parser.parse_args()

    path = args.scores_csv or next((p for p in DEFAULT_SCORE_FILES if os.path.exists(p)), None)
    if path is None or not os.path.exists(path):
        print("ERROR: no scores CSV found. Run tools/diagnose_validation_gap.py or", file=sys.stderr)
        print("tools/run_lab_validation.py first, or pass --scores-csv.", file=sys.stderr)
        sys.exit(1)

    reference = dict(NOTEBOOK_P_AUTISTIC)
    # full precision wins where both exist
    reference.update(load_xai10_reference())
    app_scores, column = load_app_scores(path, args.column)
    rows, missing = compare(reference, app_scores)

    print(f"Scores from {path} (column '{column}')")
    print(f"{'file':<26}{'notebook':>12}{'web app':>12}{'|diff|':>11}  same side of 0.5")
    print("-" * 74)
    for r in sorted(rows, key=lambda r: r["abs_diff"], reverse=True):
        print(
            f"{r['filename']:<26}{r['notebook']:>12.6f}{r['app']:>12.6f}{r['abs_diff']:>11.6f}"
            f"  {'yes' if r['same_side'] else 'NO'}"
        )
    s = summarize(rows)
    print("-" * 74)
    if s["n"]:
        print(
            f"{s['n']} images compared | max |diff| {s['max_abs_diff']:.6f} | mean |diff| {s['mean_abs_diff']:.6f}"
        )
        print(f"All on the same side of 0.5: {'yes' if s['all_same_side'] else 'NO'}")
    if missing:
        print(f"Not in the scores file (not in the test folder): {', '.join(missing)}")
    print(VERDICT_TEXT[s["verdict"]])
    print("(Notebook values from cell 3 are printed to 6 decimals, so diffs of ~1e-6 are rounding.)")


if __name__ == "__main__":
    main()
