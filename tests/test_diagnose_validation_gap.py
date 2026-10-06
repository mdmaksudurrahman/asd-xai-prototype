"""
Tests for tools/diagnose_validation_gap.py.

A diagnostic that points the wrong way is worse than none, so the parts
that decide what it reports are tested directly: the threshold rule, the
matching logic, the preprocessing variants, and — most importantly —
that its "current app" variant is exactly the app's own prediction path.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import app as app_module  # noqa: E402
import diagnose_validation_gap as dvg  # noqa: E402


def _thesis_like_scores():
    """Scores that give exactly the thesis matrix [[138,2],[5,135]] at 0.5."""
    y_true = [0] * 140 + [1] * 140
    p = [0.1] * 138 + [0.9] * 2 + [0.1] * 5 + [0.9] * 135
    return y_true, np.array(p)


# --------------------------------------------------------------------------
# confusion_at / matching_thresholds
# --------------------------------------------------------------------------
def test_confusion_at_reproduces_thesis_matrix_for_thesis_like_scores():
    y_true, p = _thesis_like_scores()
    assert dvg.confusion_at(y_true, p, 0.5) == [[138, 2], [5, 135]]


def test_threshold_rule_is_strictly_greater_like_the_app():
    """A score of exactly 0.5 is NOT Autistic — matching the app, where
    argmax over [1-p, p] picks class 1 only when p > 0.5."""
    assert dvg.confusion_at([0, 1], [0.5, 0.5], 0.5) == [[1, 0], [1, 0]]


def test_matching_thresholds_finds_a_range_including_half():
    y_true, p = _thesis_like_scores()
    found = dvg.matching_thresholds(y_true, p)
    assert 0.5 in found
    assert 0.95 not in found
    assert 0.05 not in found


def test_matching_thresholds_empty_when_nothing_matches():
    y_true, p = _thesis_like_scores()
    p = p.copy()
    p[0] = 0.9  # one extra false positive at every threshold below 0.9
    assert 0.5 not in dvg.matching_thresholds(y_true, p)
    assert dvg.matching_thresholds([0, 1], [0.2, 0.8]) == []


# --------------------------------------------------------------------------
# most_confidently_wrong
# --------------------------------------------------------------------------
def test_worst_errors_sorted_by_distance_from_threshold():
    names = ["a", "b", "c", "d"]
    y_true = [0, 0, 1, 1]
    # all four wrong; d and b are the most confident
    p = [0.55, 0.95, 0.45, 0.02]
    errors = dvg.most_confidently_wrong(names, y_true, p)
    assert [e["filename"] for e in errors] == ["d", "b", "a", "c"]


def test_worst_errors_ignores_correct_predictions():
    errors = dvg.most_confidently_wrong(["a", "b"], [0, 1], [0.1, 0.9])
    assert errors == []


def test_worst_errors_respects_top_limit():
    names = [f"n{i}" for i in range(10)]
    errors = dvg.most_confidently_wrong(names, [0] * 10, [0.9] * 10, top=3)
    assert len(errors) == 3


def test_worst_errors_label_the_classes():
    errors = dvg.most_confidently_wrong(["a"], [0], [0.9])
    assert errors[0]["true"] == "Non_Autistic"
    assert errors[0]["pred"] == "Autistic"


# --------------------------------------------------------------------------
# size_summary
# --------------------------------------------------------------------------
def test_size_summary_counts_sizes_and_native_fraction(tmp_path):
    for i, size in enumerate([(224, 224), (224, 224), (500, 400)]):
        Image.new("RGB", size).save(tmp_path / f"img{i}.png")
    paths = sorted(str(p) for p in tmp_path.glob("*.png"))
    common, native, distinct = dvg.size_summary(paths, (224, 224))
    assert distinct == 2
    assert abs(native - 2 / 3) < 1e-9
    assert common[0] == ((224, 224), 2)


# --------------------------------------------------------------------------
# preprocess_variant
# --------------------------------------------------------------------------
def _checker(tmp_path, size=120):
    """High-frequency image, so different resize methods genuinely differ."""
    rng = np.random.default_rng(0)
    path = tmp_path / "noise.png"
    Image.fromarray((rng.random((size, size, 3)) * 255).astype("uint8")).save(path)
    return str(path)


def test_every_variant_gives_right_shape_and_range(tmp_path):
    path = _checker(tmp_path)
    for name in dvg.VARIANT_ORDER:
        arr = dvg.preprocess_variant(path, name, 96, 96)
        assert arr.shape == (96, 96, 3), name
        assert arr.dtype == np.float32, name
        assert arr.min() >= -1e-6 and arr.max() <= 1 + 1e-6, name


def test_nearest_and_bilinear_genuinely_differ(tmp_path):
    path = _checker(tmp_path)
    nearest = dvg.preprocess_variant(path, "pil_nearest", 96, 96)
    bilinear = dvg.preprocess_variant(path, "pil_bilinear", 96, 96)
    assert not np.allclose(nearest, bilinear)


def test_unknown_variant_raises(tmp_path):
    try:
        dvg.preprocess_variant(_checker(tmp_path), "nonsense", 96, 96)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_nearest_variant_matches_the_apps_own_preprocessing(loaded_app, tmp_path):
    path = _checker(tmp_path)
    ours = dvg.preprocess_variant(path, "pil_nearest", app_module._img_w, app_module._img_h)
    with open(path, "rb") as f:
        _, theirs = app_module.load_preprocess(f.read())
    assert np.array_equal(ours, theirs)


# --------------------------------------------------------------------------
# score_variant: the baseline must BE the app
# --------------------------------------------------------------------------
def test_baseline_scores_equal_the_apps_own_predictions(loaded_app, synthetic_test_images):
    import os

    paths = sorted(os.path.join(synthetic_test_images, f) for f in os.listdir(synthetic_test_images))
    scores = dvg.score_variant(paths, dvg.BASELINE, app_module._img_w, app_module._img_h)
    for path, score in zip(paths, scores):
        with open(path, "rb") as f:
            _, arr01 = app_module.load_preprocess(f.read())
        _, _, probs = app_module.predict(arr01)
        assert abs(score - float(probs[1])) < 1e-5


# --------------------------------------------------------------------------
# run_diagnostics end to end
# --------------------------------------------------------------------------
def test_run_diagnostics_covers_all_variants_and_writes_scores(loaded_app, synthetic_test_images, tmp_path):
    result = dvg.run_diagnostics(synthetic_test_images, str(tmp_path), log=lambda *_: None)

    assert set(result["variants"]) == set(dvg.VARIANT_ORDER)
    assert result["n"] == 2
    for v in result["variants"].values():
        assert len(v["scores"]) == 2
        assert len(v["confusion"]) == 2

    with open(tmp_path / "diagnose_scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert set(rows[0]) == {"filename", "true_label", *dvg.VARIANT_ORDER}


def test_run_diagnostics_respects_limit(loaded_app, synthetic_test_images, tmp_path):
    result = dvg.run_diagnostics(synthetic_test_images, str(tmp_path), limit=1, log=lambda *_: None)
    assert result["n"] == 1


def test_run_diagnostics_requires_loaded_model(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_model", None)
    try:
        dvg.run_diagnostics(str(tmp_path), str(tmp_path), log=lambda *_: None)
    except RuntimeError as exc:
        assert "not loaded" in str(exc)
        return
    raise AssertionError("expected RuntimeError")


def test_print_report_runs_on_a_real_result(loaded_app, synthetic_test_images, tmp_path):
    result = dvg.run_diagnostics(synthetic_test_images, str(tmp_path), log=lambda *_: None)
    lines = []
    dvg.print_report(result, log=lines.append)
    text = "\n".join(lines)
    assert "IMAGE SIZES" in text
    assert "PREPROCESSING VARIANTS" in text
    assert "THRESHOLD SWEEP" in text
    assert "WHAT THIS SUGGESTS" in text
    assert "pil_nearest" in text and "tf_bilinear" in text


# --------------------------------------------------------------------------
# interpret
# --------------------------------------------------------------------------
def _fake_result(exact=(), native=0.1, thresholds=(), errors=()):
    variants = {
        name: {"exact_match": name in exact, "scores": np.array([0.5]), "confusion": [[0, 0], [0, 0]]}
        for name in dvg.VARIANT_ORDER
    }
    return {
        "variants": variants,
        "sizes": {"fraction_native": native, "common": [], "n_distinct": 1},
        "matching_thresholds": list(thresholds),
        "worst_errors": list(errors),
    }


def test_interpret_flags_a_different_resize_and_mentions_sefat():
    text = " ".join(dvg.interpret(_fake_result(exact=("pil_bilinear",))))
    assert "pil_bilinear" in text
    assert "Sefat" in text


def test_interpret_says_resize_cannot_explain_when_images_are_native_size():
    text = " ".join(dvg.interpret(_fake_result(native=1.0)))
    assert "cannot explain" in text


def test_interpret_points_to_model_or_data_when_nothing_matches():
    text = " ".join(dvg.interpret(_fake_result()))
    assert "checkpoint" in text


def test_interpret_notes_a_threshold_alone_does_not_explain_it():
    text = " ".join(dvg.interpret(_fake_result(thresholds=(0.4, 0.5))))
    assert "threshold alone" in text


def test_interpret_always_warns_against_adopting_a_match_blindly():
    for result in (
        _fake_result(),
        _fake_result(exact=("pil_bilinear",)),
        _fake_result(exact=(dvg.BASELINE,)),
    ):
        assert "Do not adopt" in " ".join(dvg.interpret(result))


def test_interpret_counts_borderline_errors():
    errors = [{"distance": 0.05}, {"distance": 0.45}]
    text = " ".join(dvg.interpret(_fake_result(errors=errors)))
    assert "1 of the 2" in text
