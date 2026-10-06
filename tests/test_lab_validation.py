"""
Tests for lab_validation.py (Part D / Task 5).

The pure functions are tested with hand-built inputs. The end-to-end
tests use the dummy model and synthetic images from conftest.py, with
face detection monkeypatched — they check the plumbing and the key
design decision (metrics include images where no face is found), not
real accuracy numbers.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module  # noqa: E402
import face_detection  # noqa: E402
import lab_validation as lv  # noqa: E402


def _thesis_labels():
    """Per-image labels that reproduce the thesis confusion matrix
    [[138, 2], [5, 135]] exactly (rows = true, cols = predicted)."""
    y_true = [0] * 140 + [1] * 140
    y_pred = [0] * 138 + [1] * 2 + [0] * 5 + [1] * 135
    p = [0.9 if p_ == 1 else 0.1 for p_ in y_pred]
    return y_true, y_pred, p


# --------------------------------------------------------------------------
# compute_metrics
# --------------------------------------------------------------------------
def test_metrics_reproduce_thesis_numbers_from_thesis_confusion_matrix():
    """Guards the metric definitions themselves: positive class =
    Autistic, confusion rows = true class. If any of that were mixed up,
    the thesis' own confusion matrix would not give the thesis' numbers."""
    y_true, y_pred, p = _thesis_labels()
    m = lv.compute_metrics(y_true, y_pred, p)
    assert m["n"] == 280
    assert m["confusion_matrix"] == [[138, 2], [5, 135]]
    assert round(m["accuracy"] * 100, 2) == 97.50
    assert round(m["precision"] * 100, 2) == 98.54
    assert round(m["recall"] * 100, 2) == 96.43


def test_auc_perfect_separation_is_one():
    m = lv.compute_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auc"] == 1.0


def test_auc_is_none_when_only_one_class_present():
    m = lv.compute_metrics([1, 1, 1], [1, 1, 0], [0.9, 0.8, 0.4])
    assert m["auc"] is None  # a tiny --limit run must not crash


def test_metrics_with_no_images_raises():
    try:
        lv.compute_metrics([], [], [])
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty run")


def test_precision_is_zero_not_a_crash_when_nothing_predicted_positive():
    m = lv.compute_metrics([0, 1], [0, 0], [0.1, 0.2])
    assert m["precision"] == 0.0


# --------------------------------------------------------------------------
# compare_to_thesis
# --------------------------------------------------------------------------
def test_comparison_all_match_for_thesis_confusion_matrix():
    y_true, y_pred, p = _thesis_labels()
    m = lv.compute_metrics(y_true, y_pred, p)
    # AUC needs real probabilities; the other metrics come from the matrix
    m["auc"] = 0.9914
    cmp_ = lv.compare_to_thesis(m)
    assert cmp_["comparable"] is True
    assert cmp_["confusion"]["status"] == "MATCH"
    assert all(r["status"] == "MATCH" for r in cmp_["metrics"])
    assert cmp_["all_match"] is True


def test_comparison_flags_a_difference():
    y_true, y_pred, p = _thesis_labels()
    y_pred = list(y_pred)
    y_pred[0] = 1  # one extra false positive
    m = lv.compute_metrics(y_true, y_pred, p)
    m["auc"] = 0.9914
    cmp_ = lv.compare_to_thesis(m)
    assert cmp_["confusion"]["status"] == "DIFF"
    assert cmp_["all_match"] is False
    statuses = {r["metric"]: r["status"] for r in cmp_["metrics"]}
    assert statuses["accuracy"] == "DIFF"


def test_comparison_not_comparable_for_partial_run():
    m = lv.compute_metrics([0, 1], [0, 1], [0.1, 0.9])
    cmp_ = lv.compare_to_thesis(m)
    assert cmp_["comparable"] is False
    assert cmp_["all_match"] is False


def test_comparison_auc_none_gives_na_not_crash():
    m = lv.compute_metrics([1, 1], [1, 1], [0.9, 0.8])
    cmp_ = lv.compare_to_thesis(m)
    auc_row = next(r for r in cmp_["metrics"] if r["metric"] == "auc")
    assert auc_row["status"] == "n/a"


# --------------------------------------------------------------------------
# summarize_attention
# --------------------------------------------------------------------------
def _row(name, true, pred, on_face, status="ok"):
    return {
        "filename": name,
        "true_label": true,
        "pred_label": pred,
        "correct": true == pred,
        "on_face_pct": on_face,
        "face_status": status,
    }


def test_attention_summary_groups_and_means():
    rows = [
        _row("a1.jpg", "Autistic", "Autistic", 80.0),
        _row("a2.jpg", "Autistic", "Autistic", 60.0),
        _row("n1.jpg", "Non_Autistic", "Non_Autistic", 90.0),
        # wrong, and mostly off-face
        _row("n2.jpg", "Non_Autistic", "Autistic", 30.0),
    ]
    s = lv.summarize_attention(rows)
    assert s["n_with_face"] == 4
    assert s["by_class"]["Autistic"]["mean"] == 70.0
    assert s["by_class"]["Non_Autistic"]["mean"] == 60.0
    assert s["by_correctness"]["correct"]["n"] == 3
    assert s["by_correctness"]["wrong"]["mean"] == 30.0
    assert [r["filename"] for r in s["off_face_images"]] == ["n2.jpg"]


def test_attention_summary_excludes_no_face_images_but_lists_them():
    rows = [
        _row("ok.jpg", "Autistic", "Autistic", 75.0),
        _row("noface.jpg", "Autistic", "Autistic", None, status="no_face"),
    ]
    s = lv.summarize_attention(rows)
    assert s["n_with_face"] == 1
    assert s["n_without_face"] == 1
    assert s["overall"]["mean"] == 75.0
    assert s["no_face_images"] == [{"filename": "noface.jpg", "reason": "no_face"}]


def test_off_face_boundary_exactly_50_is_not_listed():
    rows = [_row("edge.jpg", "Autistic", "Autistic", 50.0), _row("low.jpg", "Autistic", "Autistic", 49.9)]
    s = lv.summarize_attention(rows)
    assert [r["filename"] for r in s["off_face_images"]] == ["low.jpg"]


def test_off_face_images_sorted_worst_first():
    rows = [_row("b.jpg", "Autistic", "Autistic", 40.0), _row("a.jpg", "Autistic", "Autistic", 10.0)]
    s = lv.summarize_attention(rows)
    assert [r["filename"] for r in s["off_face_images"]] == ["a.jpg", "b.jpg"]


def test_attention_summary_with_no_rows_does_not_crash():
    s = lv.summarize_attention([])
    assert s["overall"]["n"] == 0
    assert s["overall"]["mean"] is None


# --------------------------------------------------------------------------
# render_html
# --------------------------------------------------------------------------
def _report(rows=None, metrics=None):
    y_true, y_pred, p = _thesis_labels()
    m = metrics or lv.compute_metrics(y_true, y_pred, p)
    return {
        "generated_at": "2026-10-06T12:00:00",
        "model": {"file": "Xception_best.h5", "sha256": "abc123", "input_size": "224x224"},
        "metrics": m,
        "comparison": lv.compare_to_thesis(m),
        "attention": lv.summarize_attention(rows or []),
    }


def test_html_shows_metrics_and_thesis_values():
    page = lv.render_html(_report())
    assert "97.50%" in page
    assert "98.54%" in page
    assert "Xception_best.h5" in page
    assert "abc123" in page


def test_html_partial_run_shows_banner_not_verdicts():
    m = lv.compute_metrics([0, 1], [0, 1], [0.1, 0.9])
    page = lv.render_html(_report(metrics=m))
    assert "Partial run" in page
    assert "not comparable" in page


def test_html_escapes_filenames():
    rows = [_row("<script>alert(1)</script>.jpg", "Autistic", "Autistic", 10.0)]
    page = lv.render_html(_report(rows=rows))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_html_makes_no_external_requests():
    page = lv.render_html(_report())
    assert "http://" not in page and "https://" not in page


# --------------------------------------------------------------------------
# run_validation end to end (dummy model, synthetic images)
# --------------------------------------------------------------------------
def test_run_validation_writes_all_three_reports(client, loaded_app, tmp_path):
    report = lv.run_validation(loaded_app.TEST_DIR, str(tmp_path), log=lambda *_: None)

    for ext in ("csv", "json", "html"):
        assert os.path.exists(tmp_path / f"lab_validation.{ext}")
    assert report["metrics"]["n"] == 2
    assert len(report["rows"]) == 2

    saved = json.loads((tmp_path / "lab_validation.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["n"] == 2
    assert "Partial run" in (tmp_path / "lab_validation.html").read_text(encoding="utf-8")


def test_metrics_include_images_where_no_face_is_found(client, loaded_app, tmp_path, monkeypatch):
    """The key design decision: the thesis evaluated every image, so a
    missing face must NOT remove an image from the accuracy numbers."""
    monkeypatch.setattr(face_detection, "detect_faces", lambda rgb01: ([], False))
    report = lv.run_validation(loaded_app.TEST_DIR, str(tmp_path), log=lambda *_: None)

    assert report["metrics"]["n"] == 2  # both images still evaluated
    assert report["attention"]["n_with_face"] == 0
    assert report["attention"]["n_without_face"] == 2
    assert all(r["on_face_pct"] is None for r in report["rows"])
    assert all(r["face_status"] == "no_face" for r in report["rows"])


def test_run_validation_attention_available_when_face_found(client, loaded_app, tmp_path):
    report = lv.run_validation(loaded_app.TEST_DIR, str(tmp_path), log=lambda *_: None)
    assert report["attention"]["n_with_face"] == 2
    assert all(0.0 <= r["on_face_pct"] <= 100.0 for r in report["rows"])


def test_run_validation_uses_the_apps_own_predictions(client, loaded_app, tmp_path):
    """Metrics must come from the same load_preprocess()+predict() the
    app uses — compare against calling them directly."""
    report = lv.run_validation(loaded_app.TEST_DIR, str(tmp_path), log=lambda *_: None)
    for row in report["rows"]:
        path = os.path.join(loaded_app.TEST_DIR, row["filename"])
        with open(path, "rb") as f:
            _, arr01 = app_module.load_preprocess(f.read())
        pred_id, _, probs = app_module.predict(arr01)
        assert row["pred_label"] == app_module.ID_TO_LABEL[pred_id]
        assert abs(row["p_autistic"] - float(probs[1])) < 1e-5


def test_run_validation_skips_unlabelled_files(client, loaded_app, tmp_path):
    from PIL import Image

    d = tmp_path / "imgs"
    d.mkdir()
    rng = np.random.default_rng(0)
    for name in ("Autistic (1).jpg", "mystery_photo.jpg"):
        Image.fromarray((rng.random((60, 60, 3)) * 255).astype("uint8")).save(d / name)

    report = lv.run_validation(str(d), str(tmp_path / "out"), log=lambda *_: None)
    assert report["metrics"]["n"] == 1
    assert [s["filename"] for s in report["skipped"]] == ["mystery_photo.jpg"]


def test_run_validation_respects_limit(client, loaded_app, tmp_path):
    report = lv.run_validation(loaded_app.TEST_DIR, str(tmp_path), limit=1, log=lambda *_: None)
    assert report["metrics"]["n"] == 1


def test_run_validation_requires_a_loaded_model(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_model", None)
    try:
        lv.run_validation(str(tmp_path), str(tmp_path), log=lambda *_: None)
    except RuntimeError as exc:
        assert "not loaded" in str(exc)
        return
    raise AssertionError("expected RuntimeError")


# --------------------------------------------------------------------------
# /lab route
# --------------------------------------------------------------------------
def test_lab_route_404_with_instructions_when_no_report(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "REPORTS_DIR", str(tmp_path / "does_not_exist"))
    resp = client.get("/lab")
    assert resp.status_code == 404
    assert b"run_lab_validation.py" in resp.data


def test_lab_route_serves_generated_report(client, loaded_app, tmp_path, monkeypatch):
    lv.run_validation(loaded_app.TEST_DIR, str(tmp_path), log=lambda *_: None)
    monkeypatch.setattr(app_module, "REPORTS_DIR", str(tmp_path))
    resp = client.get("/lab")
    assert resp.status_code == 200
    assert b"Lab validation" in resp.data
    assert b"SHA-256" in resp.data


def test_lab_route_never_runs_the_batch_itself(client, tmp_path, monkeypatch):
    """The route must only serve a finished file — a 280-image batch
    inside a web request would time out."""
    monkeypatch.setattr(app_module, "REPORTS_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(lv, "run_validation", lambda *a, **kw: calls.append(1))
    client.get("/lab")
    assert calls == []


def test_html_labels_auc_in_capitals_not_Auc():
    page = lv.render_html(_report())
    assert "<td>AUC</td>" in page
    assert "<td>Auc</td>" not in page


def test_difference_never_shows_negative_zero():
    assert lv._pp(-0.000001) == "+0.00 pp"
    assert lv._pp(0.000001) == "+0.00 pp"
    assert lv._pp(-0.0475) == "-4.75 pp"


def test_html_all_match_banner_is_styled_as_success():
    y_true, y_pred, p = _thesis_labels()
    m = lv.compute_metrics(y_true, y_pred, p)
    m["auc"] = 0.9914
    page = lv.render_html(_report(metrics=m))
    assert "note good" in page


def test_run_validation_gives_a_clear_error_when_nothing_is_evaluable(client, loaded_app, tmp_path):
    """Pointing at a folder whose filenames carry no label must produce a
    readable RuntimeError (which the CLI reports cleanly), not a bare
    ValueError traceback from deep inside the metrics code."""
    from PIL import Image

    d = tmp_path / "unlabelled"
    d.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((60, 60, 3)) * 255).astype("uint8")).save(d / "holiday_photo.jpg")

    try:
        lv.run_validation(str(d), str(tmp_path / "out"), log=lambda *_: None)
    except RuntimeError as exc:
        assert "could be evaluated" in str(exc)
        return
    raise AssertionError("expected RuntimeError")
