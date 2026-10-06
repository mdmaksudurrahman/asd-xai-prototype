"""Tests for tools/compare_with_notebook_outputs.py — no model needed."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import compare_with_notebook_outputs as cmpn  # noqa: E402


def test_reference_values_are_probabilities():
    assert len(cmpn.NOTEBOOK_P_AUTISTIC) >= 10
    assert all(0.0 <= p <= 1.0 for p in cmpn.NOTEBOOK_P_AUTISTIC.values())


def test_notebook_marks_109_and_114_as_wrong_predictions():
    """Both are errors in the notebook's own run: 109 is a Non-Autistic
    image scored > 0.5, 114 an Autistic image scored < 0.5."""
    assert cmpn.NOTEBOOK_P_AUTISTIC["Non-Autistic (109).jpg"] > 0.5
    assert cmpn.NOTEBOOK_P_AUTISTIC["Autistic (114).jpg"] < 0.5


def test_xai10_reference_loads_with_full_precision():
    ref = cmpn.load_xai10_reference()
    assert len(ref) == 10
    assert abs(ref["Autistic (10).jpg"] - 0.9980183839797974) < 1e-15
    # Non-Autistic images are stored with p1 small, i.e. p(Autistic), not confidence
    assert ref["Non-Autistic (7).jpg"] < 0.1


def test_compare_identical_scores_is_equivalent():
    ref = {"a.jpg": 0.9, "b.jpg": 0.1}
    rows, missing = cmpn.compare(ref, dict(ref))
    assert missing == []
    s = cmpn.summarize(rows)
    assert s["max_abs_diff"] == 0.0
    assert s["verdict"] == "equivalent"
    assert s["all_same_side"] is True


def test_compare_flags_small_and_large_differences():
    ref = {"a.jpg": 0.9}
    assert cmpn.summarize(cmpn.compare(ref, {"a.jpg": 0.9005})[0])["verdict"] == "close"
    assert cmpn.summarize(cmpn.compare(ref, {"a.jpg": 0.6})[0])["verdict"] == "different"


def test_compare_detects_a_flip_across_the_threshold():
    rows, _ = cmpn.compare({"a.jpg": 0.55}, {"a.jpg": 0.45})
    assert cmpn.summarize(rows)["all_same_side"] is False


def test_compare_reports_missing_images_without_failing():
    rows, missing = cmpn.compare({"a.jpg": 0.9, "valid_only.jpg": 0.1}, {"a.jpg": 0.9})
    assert [r["filename"] for r in rows] == ["a.jpg"]
    assert missing == ["valid_only.jpg"]


def test_summarize_with_nothing_compared():
    assert cmpn.summarize([])["verdict"] == "none"


def test_load_app_scores_autodetects_either_report_format(tmp_path):
    diag = tmp_path / "diag.csv"
    diag.write_text(
        "filename,true_label,pil_nearest,pil_bilinear\na.jpg,Autistic,0.8,0.7\n", encoding="utf-8"
    )
    scores, col = cmpn.load_app_scores(str(diag))
    assert col == "pil_nearest" and scores == {"a.jpg": 0.8}

    lab = tmp_path / "lab.csv"
    lab.write_text("filename,true_label,p_autistic\na.jpg,Autistic,0.6\n", encoding="utf-8")
    scores, col = cmpn.load_app_scores(str(lab))
    assert col == "p_autistic" and scores == {"a.jpg": 0.6}


def test_load_app_scores_rejects_a_file_without_a_score_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("filename,other\na.jpg,1\n", encoding="utf-8")
    try:
        cmpn.load_app_scores(str(bad))
    except ValueError as exc:
        assert "no score column" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_end_to_end_against_scores_that_equal_the_notebook(tmp_path, capsys, monkeypatch):
    reference = dict(cmpn.NOTEBOOK_P_AUTISTIC)
    reference.update(cmpn.load_xai10_reference())
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "p_autistic"])
        for name, p in reference.items():
            w.writerow([name, "x", f"{p:.6f}"])
    monkeypatch.setattr(sys, "argv", ["prog", "--scores-csv", str(path)])
    cmpn.main()
    out = capsys.readouterr().out
    assert "within floating-point noise" in out
    assert "Non-Autistic (109).jpg" in out
