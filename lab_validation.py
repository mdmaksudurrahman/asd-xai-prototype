# -*- coding: utf-8 -*-
"""
Part D / Task 5: lab validation.

Runs the real test set through the web app's own code and compares the
result with the numbers published in the thesis — the evidence that the
web app and the thesis model are the same thing.

Two deliberately different paths, because they answer different questions:

  1. METRICS (accuracy, precision, recall, AUC, confusion matrix) use the
     thesis-faithful path: app.load_preprocess() -> app.predict() on ALL
     images, with NO face-check gating. The thesis evaluated every test
     image, so gating (which rejects images where no face is detected)
     would make the numbers incomparable.

  2. ATTENTION-ON-FACE statistics use the app's own pipeline
     (face check -> explain_image), so they only exist for images where a
     face was found. The report states how many images that is, and lists
     the ones that were left out.

The pure functions (compute_metrics, compare_to_thesis,
summarize_attention, render_html) need no model, detector or image, and
are unit-tested in tests/test_lab_validation.py. run_validation() is the
I/O-heavy loop around them.

Reports contain filenames and numbers only — never images.
"""

import csv
import datetime
import hashlib
import html
import json
import os
import statistics
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

import app as app_module
import region_analysis
import wording

# Numbers published in the thesis (positive class = Autistic; confusion
# matrix rows = true class, columns = predicted class, order
# [Non-Autistic, Autistic]).
THESIS_REFERENCE = {
    "accuracy": 0.9750,
    "precision": 0.9854,
    "recall": 0.9643,
    "auc": 0.9914,
    "confusion_matrix": [[138, 2], [5, 135]],
    "n_images": 280,
}

# The thesis figures are rounded to 2 decimal places of a percentage, so
# anything within 0.01 percentage points is the same number.
METRIC_TOLERANCE = 0.0001

METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "auc": "AUC"}

OFF_FACE_THRESHOLD_PCT = region_analysis.ON_FACE_WARNING_THRESHOLD * 100

REPORT_BASENAME = "lab_validation"


# --------------------------------------------------------------------------
# Pure functions (unit-tested without a model, detector or image)
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, p_autistic):
    """y_true / y_pred: 0 = Non-Autistic, 1 = Autistic. p_autistic: the
    model's probability for class 1 (used only for AUC)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p = np.asarray(p_autistic)
    if len(y_true) == 0:
        raise ValueError("No images to compute metrics on.")

    # AUC is undefined with only one class present (e.g. a tiny --limit
    # run). Depending on the scikit-learn version that either raises or
    # silently returns nan, so check explicitly rather than relying on
    # either behaviour.
    auc = None
    if len(np.unique(y_true)) == 2:
        value = float(roc_auc_score(y_true, p))
        auc = value if np.isfinite(value) else None

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def compare_to_thesis(metrics, tol=METRIC_TOLERANCE, thesis=None):
    """Per-metric comparison against the thesis numbers.

    `comparable` is False unless the run covered the same number of
    images as the thesis (280) — a partial run's metrics can't be
    compared, and the report says so instead of showing misleading
    MATCH/DIFF verdicts as if they meant something.
    """
    thesis = thesis or THESIS_REFERENCE
    rows = []
    for name in ("accuracy", "precision", "recall", "auc"):
        ours = metrics.get(name)
        ref = thesis[name]
        if ours is None:
            rows.append({"metric": name, "ours": None, "thesis": ref, "diff": None, "status": "n/a"})
            continue
        diff = ours - ref
        rows.append(
            {
                "metric": name,
                "ours": ours,
                "thesis": ref,
                "diff": diff,
                "status": "MATCH" if abs(diff) <= tol else "DIFF",
            }
        )

    cm_match = metrics["confusion_matrix"] == thesis["confusion_matrix"]
    comparable = metrics["n"] == thesis["n_images"]
    all_match = comparable and cm_match and all(r["status"] == "MATCH" for r in rows)

    return {
        "metrics": rows,
        "confusion": {
            "ours": metrics["confusion_matrix"],
            "thesis": thesis["confusion_matrix"],
            "status": "MATCH" if cm_match else "DIFF",
        },
        "comparable": comparable,
        "all_match": all_match,
    }


def _stats(values):
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
    }


def summarize_attention(rows, threshold_pct=OFF_FACE_THRESHOLD_PCT):
    """Attention-on-face statistics from per-image rows. Each row needs:
    filename, true_label, pred_label, correct, on_face_pct (None when the
    face check found no usable face), face_status."""
    with_face = [r for r in rows if r.get("on_face_pct") is not None]
    without_face = [r for r in rows if r.get("on_face_pct") is None]

    by_class = {
        label: _stats([r["on_face_pct"] for r in with_face if r["true_label"] == label])
        for label in ("Autistic", "Non_Autistic")
    }
    by_correctness = {
        "correct": _stats([r["on_face_pct"] for r in with_face if r["correct"]]),
        "wrong": _stats([r["on_face_pct"] for r in with_face if not r["correct"]]),
    }

    off_face = sorted(
        (
            {
                "filename": r["filename"],
                "true_label": r["true_label"],
                "pred_label": r["pred_label"],
                "on_face_pct": r["on_face_pct"],
            }
            for r in with_face
            if r["on_face_pct"] < threshold_pct
        ),
        key=lambda r: r["on_face_pct"],
    )

    return {
        "n_with_face": len(with_face),
        "n_without_face": len(without_face),
        "overall": _stats([r["on_face_pct"] for r in with_face]),
        "by_class": by_class,
        "by_correctness": by_correctness,
        "off_face_threshold_pct": threshold_pct,
        "off_face_images": off_face,
        "no_face_images": [{"filename": r["filename"], "reason": r.get("face_status")} for r in without_face],
    }


def _pct(v, digits=2):
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def _pp(v):
    if v is None:
        return "n/a"
    x = round(v * 100, 2)
    if x == 0:  # avoid "-0.00 pp" from floating-point noise
        x = 0.0
    return f"{x:+.2f} pp"


def _badge(status):
    css = {"MATCH": "ok", "DIFF": "bad"}.get(status, "na")
    return f'<span class="badge {css}">{html.escape(status)}</span>'


def render_html(report):
    """Standalone HTML page (inline CSS, no external requests) for a
    report dict as built by run_validation()."""
    esc = html.escape
    cmp_ = report["comparison"]
    m = report["metrics"]
    att = report["attention"]
    model = report["model"]

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Lab validation — web app vs thesis</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1d2530;line-height:1.5}",
        "h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #d5dae0;padding-bottom:.3rem}",
        "table{border-collapse:collapse;margin:.5rem 0 1rem}th,td{border:1px solid #d5dae0;padding:.35rem .7rem;text-align:right}",
        "th:first-child,td:first-child{text-align:left}th{background:#f3f5f7}",
        ".badge{padding:.1rem .5rem;border-radius:999px;font-size:.8rem;font-weight:600}",
        ".ok{background:#d7f2de;color:#14532d}.bad{background:#fde0de;color:#7f1d1d}.na{background:#e5e7eb;color:#374151}",
        ".note{background:#fff7e0;border:1px solid #f0d58a;padding:.6rem .9rem;border-radius:6px}",
        ".good{background:#e6f6ea;border-color:#9bd3a8}",
        "table.txt th,table.txt td{text-align:left}",
        ".small{font-size:.85rem;color:#4b5563}code{background:#f3f5f7;padding:0 .25rem}",
        ".research-banner{margin:-2rem -1rem 1.5rem;padding:.55rem 1rem;background:#fff4d6;color:#5c3b00;border-bottom:1px solid #e6c97a;font-weight:500;font-size:.875rem;text-align:center}",
        "</style></head><body>",
        f"<div class='research-banner' role='note'>{esc(wording.RESEARCH_BANNER)}</div>",
        "<h1>Lab validation — web app vs thesis</h1>",
        "<p class='small'>Research prototype output. Generated "
        f"{esc(report['generated_at'])} from {m['n']} images.</p>",
    ]

    if not cmp_["comparable"]:
        parts.append(
            f"<p class='note'><strong>Partial run</strong> ({m['n']} of "
            f"{THESIS_REFERENCE['n_images']} images). These numbers are not comparable to the thesis — "
            "run the full test set for a MATCH/DIFF verdict.</p>"
        )
    elif cmp_["all_match"]:
        parts.append(
            "<p class='note good'>All published thesis metrics and the confusion matrix are reproduced "
            "by the web app's code path.</p>"
        )

    parts.append("<h2>Classification metrics (all images, no face-check gating)</h2>")
    parts.append(
        "<table><tr><th>Metric</th><th>Web app</th><th>Thesis</th><th>Difference</th><th>Verdict</th></tr>"
    )
    for r in cmp_["metrics"]:
        verdict = (
            _badge(r["status"]) if cmp_["comparable"] else "<span class='badge na'>not comparable</span>"
        )
        parts.append(
            f"<tr><td>{esc(METRIC_LABELS[r['metric']])}</td><td>{_pct(r['ours'])}</td>"
            f"<td>{_pct(r['thesis'])}</td><td>{_pp(r['diff'])}</td><td>{verdict}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h2>Confusion matrix</h2>")
    parts.append("<p class='small'>Rows = true class, columns = predicted class.</p>")
    for title, matrix in (("Web app", cmp_["confusion"]["ours"]), ("Thesis", cmp_["confusion"]["thesis"])):
        parts.append(
            f"<table><caption class='small'>{esc(title)}</caption>"
            "<tr><th></th><th>Pred. Non-Autistic</th><th>Pred. Autistic</th></tr>"
            f"<tr><td>True Non-Autistic</td><td>{matrix[0][0]}</td><td>{matrix[0][1]}</td></tr>"
            f"<tr><td>True Autistic</td><td>{matrix[1][0]}</td><td>{matrix[1][1]}</td></tr></table>"
        )
    if cmp_["comparable"]:
        parts.append(f"<p>Confusion matrix: {_badge(cmp_['confusion']['status'])}</p>")

    parts.append("<h2>Attention on the face (Grad-CAM)</h2>")
    parts.append(
        f"<p class='small'>Share of Grad-CAM attention inside the detected face box (approximate). "
        f"Computed for the {att['n_with_face']} images where a face was found; "
        f"{att['n_without_face']} images with no usable face are excluded here "
        "(but still counted in the metrics above, as in the thesis).</p>"
    )

    def stat_row(label, s):
        mean = "n/a" if s["mean"] is None else f"{s['mean']:.1f}%"
        median = "n/a" if s["median"] is None else f"{s['median']:.1f}%"
        return f"<tr><td>{esc(label)}</td><td>{s['n']}</td><td>{mean}</td><td>{median}</td></tr>"

    parts.append("<table><tr><th>Group</th><th>n</th><th>Mean on-face</th><th>Median on-face</th></tr>")
    parts.append(stat_row("All images with a face", att["overall"]))
    parts.append(stat_row("True class: Autistic", att["by_class"]["Autistic"]))
    parts.append(stat_row("True class: Non-Autistic", att["by_class"]["Non_Autistic"]))
    parts.append(stat_row("Correct predictions", att["by_correctness"]["correct"]))
    parts.append(stat_row("Wrong predictions", att["by_correctness"]["wrong"]))
    parts.append("</table>")

    parts.append(
        f"<h2>Images where attention is mostly off the face (&lt; {att['off_face_threshold_pct']:.0f}%)</h2>"
    )
    if att["off_face_images"]:
        parts.append(
            "<table class='txt'><tr><th>File</th><th>True</th><th>Predicted</th><th>On-face</th></tr>"
        )
        for r in att["off_face_images"]:
            parts.append(
                f"<tr><td>{esc(r['filename'])}</td><td>{esc(r['true_label'])}</td>"
                f"<td>{esc(r['pred_label'])}</td><td>{r['on_face_pct']:.1f}%</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p>None.</p>")

    parts.append("<h2>Images with no usable face (excluded from attention statistics)</h2>")
    if att["no_face_images"]:
        parts.append("<table class='txt'><tr><th>File</th><th>Reason</th></tr>")
        for r in att["no_face_images"]:
            parts.append(f"<tr><td>{esc(r['filename'])}</td><td>{esc(str(r['reason']))}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p>None.</p>")

    parts.append("<h2>Provenance</h2>")
    parts.append(
        f"<p class='small'>Model file: <code>{esc(model['file'])}</code><br>"
        f"SHA-256: <code>{esc(model['sha256'])}</code><br>"
        f"Input size: {esc(str(model['input_size']))}<br>"
        "Metrics preprocessing: nearest-neighbour resize, pixel values / 255, as in the thesis notebook.</p>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def sha256_of_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _list_images(images_dir):
    return sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in app_module.ALLOWED_EXT
    )


def run_validation(images_dir, out_dir, limit=None, log=print):
    """Run every image in images_dir and write lab_validation.{csv,json,html}
    into out_dir. The model must already be loaded (app.load_asd_model()).
    Returns the report dict."""
    if not app_module.model_ready():
        raise RuntimeError("Model is not loaded — call app.load_asd_model() first.")

    paths = _list_images(images_dir)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"No images found in {images_dir}")

    rows = []
    skipped = []
    y_true, y_pred, p_autistic = [], [], []
    started = time.time()

    for i, path in enumerate(paths, 1):
        fname = os.path.basename(path)
        true_label = app_module.infer_true_label_from_filename(fname)
        if true_label is None:
            skipped.append({"filename": fname, "reason": "label not recognisable from filename"})
            log(f"  [{i}/{len(paths)}] {fname}: cannot infer label from filename, skipping")
            continue

        try:
            with open(path, "rb") as f:
                image_bytes = f.read()

            # 1) Thesis-faithful path: same preprocessing as the notebook,
            #    every image, no face-check gating.
            _, arr01 = app_module.load_preprocess(image_bytes)
            pred_id, conf, probs = app_module.predict(arr01)

            # 2) The app's own pipeline, for the attention-on-face numbers.
            on_face_pct, strongest, face_status = None, None, "ok"
            face_check = app_module.run_face_check_on_bytes(image_bytes)
            if face_check["ok"]:
                result = app_module.explain_image(
                    image_bytes,
                    mode="quick",
                    crop_box=face_check["crop_box"],
                    face_box_224=face_check["face_box_224"],
                    landmarks_224=face_check["landmarks_224"],
                    blur_eyes=False,
                )
                gradcam = result["region_attention"]["gradcam"]
                on_face_pct = region_analysis.on_face_percentage(gradcam["percentages"])
                strongest = gradcam["strongest_region"]
            else:
                face_status = face_check["reason"]
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a 280-image run
            skipped.append({"filename": fname, "reason": f"error: {exc}"})
            log(f"  [{i}/{len(paths)}] {fname}: error ({exc}), skipping")
            continue

        true_id = 1 if true_label == "Autistic" else 0
        y_true.append(true_id)
        y_pred.append(pred_id)
        p_autistic.append(float(probs[1]))
        rows.append(
            {
                "filename": fname,
                "true_label": true_label,
                "pred_label": app_module.ID_TO_LABEL[pred_id],
                "correct": bool(pred_id == true_id),
                "p_autistic": round(float(probs[1]), 6),
                "confidence": round(conf, 6),
                "face_status": face_status,
                "on_face_pct": on_face_pct,
                "strongest_region": strongest,
            }
        )

        if i % 10 == 0 or i == len(paths):
            elapsed = time.time() - started
            eta = elapsed / i * (len(paths) - i)
            log(f"  [{i}/{len(paths)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    if not rows:
        raise RuntimeError(
            f"None of the {len(paths)} images in {images_dir} could be evaluated "
            "(no recognisable 'Autistic' / 'Non-Autistic' label in the filenames, or all failed to load)."
        )

    metrics = compute_metrics(y_true, y_pred, p_autistic)
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": {
            "file": os.path.basename(app_module.MODEL_PATH),
            "sha256": sha256_of_file(app_module.MODEL_PATH),
            "input_size": f"{app_module._img_w}x{app_module._img_h}",
        },
        "n_images_found": len(paths),
        "skipped": skipped,
        "metrics": metrics,
        "comparison": compare_to_thesis(metrics),
        "attention": summarize_attention(rows),
        "rows": rows,
    }

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, REPORT_BASENAME)

    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(base + ".html", "w", encoding="utf-8") as f:
        f.write(render_html(report))

    return report
