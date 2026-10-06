"""Tests for model_card.py — pure file/JSON logic, no model or Flask needed."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import model_card as mc  # noqa: E402

THE_SHA = "a" * 64
OTHER_SHA = "b" * 64


def _report(sha=THE_SHA, n=280, auc=0.9916, with_thesis_block=True):
    report = {
        "generated_at": "2026-10-06T17:41:00",
        "model": {"file": "Xception_best.h5", "sha256": sha, "input_size": "224x224"},
        "metrics": {
            "n": n,
            "accuracy": 0.9607,
            "precision": 0.9510,
            "recall": 0.9714,
            "auc": auc,
            "confusion_matrix": [[133, 7], [4, 136]],
        },
    }
    if with_thesis_block:
        # run_validation() writes this block; the About page must never use it
        report["comparison"] = {
            "metrics": [{"metric": "accuracy", "ours": 0.9607, "thesis": 0.975, "status": "DIFF"}],
            "confusion": {"ours": [[133, 7], [4, 136]], "thesis": [[138, 2], [5, 135]], "status": "DIFF"},
            "comparable": True,
            "all_match": False,
        }
    return report


def _write(tmp_path, report, name="lab_validation.json"):
    path = tmp_path / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------
# file_sha256 / file_size_mb
# --------------------------------------------------------------------------
def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "model.h5"
    f.write_bytes(b"pretend model weights" * 1000)
    assert mc.file_sha256(str(f)) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_sha256_none_for_missing_file_or_path():
    assert mc.file_sha256("/definitely/not/here.h5") is None
    assert mc.file_sha256(None) is None


def test_sha256_is_cached_between_calls(tmp_path, monkeypatch):
    f = tmp_path / "model.h5"
    f.write_bytes(b"x" * 100)
    calls = []
    real = mc._hash_file
    monkeypatch.setattr(mc, "_hash_file", lambda p: calls.append(p) or real(p))
    mc.file_sha256(str(f))
    mc.file_sha256(str(f))
    assert len(calls) == 1


def test_sha256_recomputed_when_the_file_is_replaced(tmp_path):
    f = tmp_path / "model.h5"
    f.write_bytes(b"first version")
    first = mc.file_sha256(str(f))
    f.write_bytes(b"a different, longer second version")
    assert mc.file_sha256(str(f)) != first


def test_file_size_mb(tmp_path):
    f = tmp_path / "m.h5"
    f.write_bytes(b"0" * (2 * 1024 * 1024))
    assert mc.file_size_mb(str(f)) == 2.0
    assert mc.file_size_mb("/nope") is None


# --------------------------------------------------------------------------
# facts
# --------------------------------------------------------------------------
def test_facts_defaults_when_file_missing_or_corrupt(tmp_path):
    assert mc.load_facts(str(tmp_path / "nope.json")) == mc.DEFAULT_FACTS
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert mc.load_facts(str(bad)) == mc.DEFAULT_FACTS


def test_facts_file_overrides_and_ignores_null_and_underscore_keys(tmp_path):
    f = tmp_path / "facts.json"
    f.write_text(
        json.dumps(
            {
                "_help": "ignored",
                "dataset_name": "My dataset",
                "training_set_size": None,
                "intended_use": None,
                "limitations": ["Only one."],
            }
        ),
        encoding="utf-8",
    )
    facts = mc.load_facts(str(f))
    assert facts["dataset_name"] == "My dataset"
    assert facts["training_set_size"] is None
    # null -> default kept
    assert facts["intended_use"] == mc.DEFAULT_FACTS["intended_use"]
    assert facts["limitations"] == ["Only one."]
    assert "_help" not in facts


def test_the_shipped_facts_file_is_valid_and_has_only_known_keys():
    data = json.loads((ROOT / "model_card_facts.json").read_text(encoding="utf-8"))
    unknown = {k for k in data if not k.startswith("_")} - set(mc.DEFAULT_FACTS)
    assert unknown == set()


def test_placeholder_for_empty_values():
    assert mc.fact_or_placeholder(None) == mc.TO_BE_COMPLETED
    assert mc.fact_or_placeholder("") == mc.TO_BE_COMPLETED
    assert mc.fact_or_placeholder([]) == mc.TO_BE_COMPLETED
    assert mc.fact_or_placeholder("1,000 images") == "1,000 images"


def test_default_limitations_make_no_performance_claims():
    text = " ".join(mc.DEFAULT_FACTS["limitations"]) + mc.DEFAULT_FACTS["intended_use"]
    assert "%" not in text


# --------------------------------------------------------------------------
# load_performance
# --------------------------------------------------------------------------
def test_performance_missing_report(tmp_path):
    p = mc.load_performance(str(tmp_path / "none.json"), THE_SHA)
    assert p["state"] == "missing"
    assert "run_lab_validation" in p["message"]


def test_performance_unreadable_report(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ nope", encoding="utf-8")
    assert mc.load_performance(str(bad), THE_SHA)["state"] == "unreadable"

    no_metrics = _write(tmp_path, {"model": {"sha256": THE_SHA}}, "nometrics.json")
    assert mc.load_performance(no_metrics, THE_SHA)["state"] == "unreadable"


def test_performance_report_with_wrong_shaped_fields_is_unreadable_not_a_crash(tmp_path):
    """A hand-edited or corrupted report must give the 'could not be read'
    notice, never an exception that would turn the About page into a 500."""
    odd = _report()
    odd["model"] = "not an object"
    assert mc.load_performance(_write(tmp_path, odd, "odd1.json"), THE_SHA)["state"] == "unreadable"

    odd = _report()
    odd["metrics"] = ["not", "an", "object"]
    assert mc.load_performance(_write(tmp_path, odd, "odd2.json"), THE_SHA)["state"] == "unreadable"

    assert mc.load_performance(_write(tmp_path, ["a", "list"], "odd3.json"), THE_SHA)["state"] == "unreadable"


def test_performance_ok_for_full_run_on_the_current_model(tmp_path):
    p = mc.load_performance(_write(tmp_path, _report()), THE_SHA)
    assert p["state"] == "ok"
    assert p["n"] == 280
    rows = dict(p["rows"])
    assert rows["Accuracy"] == "96.07%"
    assert rows["AUC"] == "99.16%"
    assert p["confusion_matrix"] == [[133, 7], [4, 136]]
    assert p["message"] is None


def test_performance_partial_run_is_flagged_with_its_size(tmp_path):
    p = mc.load_performance(_write(tmp_path, _report(n=20)), THE_SHA)
    assert p["state"] == "partial"
    assert "20" in p["message"]


def test_performance_withheld_when_measured_on_a_different_model_file(tmp_path):
    p = mc.load_performance(_write(tmp_path, _report(sha=OTHER_SHA)), THE_SHA)
    assert p["state"] == "stale"
    assert "rows" not in p and "confusion_matrix" not in p


def test_performance_shown_with_a_note_when_the_model_file_is_unavailable(tmp_path):
    p = mc.load_performance(_write(tmp_path, _report()), None)
    assert p["state"] == "ok"
    assert "not available to confirm" in p["message"]


def test_performance_missing_auc_reads_not_available(tmp_path):
    p = mc.load_performance(_write(tmp_path, _report(auc=None)), THE_SHA)
    assert dict(p["rows"])["AUC"] == "not available"


def test_performance_never_exposes_any_comparison_or_thesis_figure(tmp_path):
    """The report on disk contains a thesis-comparison block. Nothing from
    it may reach the About page's data."""
    p = mc.load_performance(_write(tmp_path, _report(with_thesis_block=True)), THE_SHA)
    text = json.dumps(p, default=str)
    assert "thesis" not in text.lower()
    assert "comparison" not in text.lower()
    for forbidden in ("0.975", "97.5", "138", "MATCH", "DIFF"):  # case-sensitive on purpose
        assert forbidden not in text, forbidden
