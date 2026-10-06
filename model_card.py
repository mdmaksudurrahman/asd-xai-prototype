# -*- coding: utf-8 -*-
"""
Part D / Task 6: the model card behind the About page.

Everything here is pure file/JSON handling with no TensorFlow import, so
it is fast to test and can't create an import cycle with app.py.

Three principles:

  1. NUMBERS COME FROM A MEASUREMENT, NOT FROM TYPING. Performance figures
     are read from reports/lab_validation.json, written by
     tools/run_lab_validation.py. Only what this application measured is
     shown here — no comparison with any other source.

  2. NEVER SHOW STALE FIGURES. The report records the SHA-256 of the model
     file it was measured on. If the model file has since changed, the
     figures are withheld and the page says so, instead of quietly
     presenting numbers that belong to a different model.

  3. DON'T INVENT FACTS. Training details this code can't know (dataset
     size, augmentation, ...) live in model_card_facts.json and show as
     "To be completed" until someone fills them in.
"""

import hashlib
import json
import os

# Number of images in the held-out test set. A report covering fewer is
# labelled as a partial run rather than presented as the full result.
FULL_TEST_SET_SIZE = 280

TO_BE_COMPLETED = "To be completed by the research team."

DEFAULT_FACTS = {
    "dataset_name": None,
    "dataset_source": None,
    "training_set_size": None,
    "class_balance": None,
    "data_split": None,
    "augmentation": None,
    "training_notes": None,
    "intended_use": (
        "Research and demonstration of explainable AI for facial-image analysis. "
        "Not for clinical, diagnostic or screening use, and not for decisions about any individual."
    ),
    "limitations": [
        "Performance was measured on a single held-out test set; how it generalises to other "
        "populations, ages, cameras or lighting has not been tested.",
        "The model's confidence is not a calibrated probability of being correct.",
        "Photos without exactly one clear, large enough face are rejected before analysis.",
        "Explanation maps (Grad-CAM, Score-CAM, LIME) show where the model looked, not why a "
        "condition is or is not present. Facial regions are approximate.",
    ],
}


# --------------------------------------------------------------------------
# Model file
# --------------------------------------------------------------------------
_sha_cache = {}


def _hash_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_sha256(path):
    """SHA-256 of a file, cached by (path, modification time, size) so a
    page view doesn't re-hash a ~90 MB model every time, but a replaced
    file is re-hashed. None if the file doesn't exist."""
    if not path or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
    if key not in _sha_cache:
        _sha_cache[key] = _hash_file(path)
    return _sha_cache[key]


def file_size_mb(path):
    if not path or not os.path.isfile(path):
        return None
    return round(os.path.getsize(path) / (1024 * 1024), 1)


# --------------------------------------------------------------------------
# Editable facts
# --------------------------------------------------------------------------
def load_facts(path):
    """DEFAULT_FACTS overlaid with whatever model_card_facts.json provides.
    A missing or unreadable file just gives the defaults."""
    facts = dict(DEFAULT_FACTS)
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, ValueError):
        return facts
    if isinstance(loaded, dict):
        for key in DEFAULT_FACTS:
            if key in loaded and loaded[key] not in (None, "", []):
                facts[key] = loaded[key]
    return facts


def fact_or_placeholder(value):
    return TO_BE_COMPLETED if value in (None, "", []) else value


# --------------------------------------------------------------------------
# Measured performance
# --------------------------------------------------------------------------
def _pct(v):
    return None if v is None else f"{v * 100:.2f}%"


def load_performance(report_path, current_model_sha256):
    """Measured performance for the About page.

    state is one of:
      "missing"    no report has been generated yet
      "unreadable" the report file exists but can't be parsed
      "stale"      measured on a different model file -> figures withheld
      "partial"    covers fewer images than the full test set (figures shown, flagged)
      "ok"         full run on the current model file

    Only what was measured is returned — nothing from any other source.
    """
    if not os.path.exists(report_path):
        return {
            "state": "missing",
            "message": "Performance figures have not been generated yet. "
            "Run tools/run_lab_validation.py to measure them on the test set.",
        }

    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        metrics = report["metrics"]
        measured_sha = report.get("model", {}).get("sha256")
        n = int(metrics["n"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {
            "state": "unreadable",
            "message": "The performance report could not be read. "
            "Re-run tools/run_lab_validation.py to regenerate it.",
        }

    if current_model_sha256 and measured_sha and measured_sha != current_model_sha256:
        return {
            "state": "stale",
            "message": "The performance figures on file were measured on a different version of "
            "the model file, so they are not shown. Re-run tools/run_lab_validation.py.",
        }

    cm = metrics.get("confusion_matrix")
    result = {
        "state": "ok" if n >= FULL_TEST_SET_SIZE else "partial",
        "message": None,
        "n": n,
        "measured_at": report.get("generated_at"),
        "rows": [
            ("Accuracy", _pct(metrics.get("accuracy"))),
            ("Precision (Autistic as positive class)", _pct(metrics.get("precision"))),
            ("Recall (Autistic as positive class)", _pct(metrics.get("recall"))),
            ("AUC", _pct(metrics.get("auc")) or "not available"),
        ],
        "confusion_matrix": cm,
        "matched_to_model_file": bool(current_model_sha256 and measured_sha),
    }
    if result["state"] == "partial":
        result["message"] = (
            f"These figures come from a partial run of {n} images (the full test set has "
            f"{FULL_TEST_SET_SIZE}), so treat them as indicative only."
        )
    elif not result["matched_to_model_file"]:
        result["message"] = "The model file was not available to confirm these figures belong to it."
    return result
