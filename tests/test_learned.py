"""Runtime model: contract validation, fallback, suppression, parity with offline windows."""
import json
import numpy as np
import pytest
from dms import learned
from dms.features import FeatureConfig, MODEL_FEATURE_NAMES, TabularPreprocessor, feature_vector, iter_window_bounds, rows_to_arrays, window_features
from dms.learned import Bundle, BundleError, LearnedStatePredictor, format_status, gru_logits
from research.bundle import write_bundle

FCFG = FeatureConfig(window_seconds=10, stride_seconds=2, min_coverage=.7)


def make_bundle(path, temperature=None, model="logreg"):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(MODEL_FEATURE_NAMES)))
    pre = TabularPreprocessor().fit(X)
    W = rng.normal(size=(3, pre.output_dim)) * .1
    logits = rng.normal(size=(30, 3))
    write_bundle(path, "logreg", {"W": W, "b": np.zeros(3)}, pre.to_dict(), FCFG, logits, rng.integers(0, 3, 30), temperature,
                 {"dataset": "unit test", "synthetic": True})
    return path


def record(valid=True, ear=.3, rng=None):
    n = lambda s=.01: (rng.normal(0, s) if rng else 0)
    nan = float("nan")
    if not valid:
        return dict(detected=False, valid=False, ear_l=nan, ear_r=nan, mar=nan, yaw=nan, pitch=nan, roll=nan)
    return dict(detected=True, valid=True, ear_l=ear + n(), ear_r=ear + n(), mar=.1 + n(), yaw=n(1), pitch=n(1), roll=n(1))


def feed(pred, seconds, fps=15, valid=lambda t: True, start=0.0, seed=1):
    rng, last = np.random.default_rng(seed), None
    for i in range(int(seconds * fps)):
        t = start + i / fps
        last = pred.update(t, record(valid(t), rng=rng))
    return last


def test_missing_bundle_and_corrupt_bundle_fall_back(tmp_path):
    assert LearnedStatePredictor.from_path(None).update(0, record())["state"] == "disabled"
    assert LearnedStatePredictor.from_path(tmp_path / "nope").update(0, record())["state"] == "disabled"
    bad = make_bundle(tmp_path / "bad")
    (bad / "weights.npz").write_bytes(b"tampered")
    p = LearnedStatePredictor.from_path(bad)
    assert p.update(0, record())["state"] == "disabled" and "hash" in p.error
    (bad / "bundle.json").write_text("{not json")
    assert LearnedStatePredictor.from_path(bad).bundle is None


def test_feature_version_mismatch_refused(tmp_path):
    b = make_bundle(tmp_path / "b")
    meta = json.loads((b / "bundle.json").read_text())
    meta["feature_version"] = "0"
    (b / "bundle.json").write_text(json.dumps(meta))
    with pytest.raises(BundleError, match="feature version"):
        Bundle(b)


def test_inference_failure_never_raises_and_disables(tmp_path, monkeypatch):
    p = LearnedStatePredictor.from_path(make_bundle(tmp_path / "b"))
    monkeypatch.setattr(Bundle, "logits_from_window", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    status = feed(p, 20)
    assert status["state"] in ("error", "disabled") and "boom" in (p.error or "")
    assert p.update(99, record())["state"] == "disabled"           # stays out of the way afterwards


def test_warmup_then_prediction_then_age(tmp_path):
    p = LearnedStatePredictor.from_path(make_bundle(tmp_path / "b"))
    assert feed(p, 5)["state"] == "warming"
    status = feed(LearnedStatePredictor.from_path(tmp_path / "b"), 14)
    assert status["state"] == "ok"
    text = format_status(status, now=status["prediction"]["window_end"] + 1.5)
    assert "Prediction age: 1.5 s" in text and "UNCALIBRATED" in text


def test_low_coverage_suppresses_prediction(tmp_path):
    p = LearnedStatePredictor.from_path(make_bundle(tmp_path / "b"))
    assert feed(p, 12)["state"] == "ok"
    status = feed(p, 12, valid=lambda t: (t % 4) < 1.0, start=12.0)         # mostly lost tracking
    assert status["state"] == "insufficient" and status["prediction"] is None
    assert "No learned class shown" in format_status(status)


def test_recent_loss_of_face_suppresses_even_with_good_window_coverage(tmp_path):
    p = LearnedStatePredictor.from_path(make_bundle(tmp_path / "b"))
    feed(p, 12)
    status = feed(p, 3, valid=lambda t: False, start=12.0)
    assert status["state"] == "insufficient"


def test_backwards_clock_resets_instead_of_stitching(tmp_path):
    p = LearnedStatePredictor.from_path(make_bundle(tmp_path / "b"))
    feed(p, 12)
    assert p.update(0.0, record())["state"] == "warming"


def test_stale_prediction_is_suppressed(tmp_path):
    p = LearnedStatePredictor.from_path(make_bundle(tmp_path / "b"), max_age_seconds=1.0)
    feed(p, 12)
    p.last_end = 1e9                                                         # block refresh, so the prediction ages
    assert feed(p, 3, start=12.0)["state"] == "stale"


def test_score_kind_labels(tmp_path):
    assert Bundle(make_bundle(tmp_path / "u")).score_kind == "uncalibrated"
    c = make_bundle(tmp_path / "c", temperature=1.7)
    assert Bundle(c).score_kind == "calibrated_unverified"
    assert "held-out check NOT recorded" in Bundle(c).score_description(.6)
    meta = json.loads((c / "bundle.json").read_text())
    meta["validation_status"]["heldout_calibration"] = {"ece": .08, "n_participants": 12}
    (c / "bundle.json").write_text(json.dumps(meta))
    b = Bundle(c)
    assert b.score_kind == "calibrated_heldout_checked" and "ECE 0.08" in b.score_description(.6)


def test_offline_window_and_live_predictor_use_identical_features(tmp_path):
    """Preprocessing parity: the live rolling buffer must reproduce the offline window exactly."""
    bundle = Bundle(make_bundle(tmp_path / "b"))
    rng = np.random.default_rng(3)
    fps, n = 15, 15 * 25
    recs = [record(valid=bool(rng.random() > .1), ear=.3 if rng.random() > .1 else .12, rng=rng) for _ in range(n)]
    rows = [(i / fps, r["valid"], r["ear_l"], r["ear_r"], r["mar"], r["yaw"], r["pitch"], r["roll"]) for i, r in enumerate(recs)]
    arrays = rows_to_arrays(rows)
    t0, t1 = list(iter_window_bounds(0.0, arrays["t"][-1], FCFG))[3]
    offline, _ = bundle.logits_from_window(arrays, t0, t1)
    live = LearnedStatePredictor(bundle)
    seen = None
    for i, r in enumerate(recs):
        t = i / fps
        status = live.update(t, r)
        if t >= t1 - 1e-9 and live.prediction and abs(live.prediction["window_end"] - t1) < 1e-6:
            seen = live.prediction
            break
    assert seen is not None
    expected = bundle.probabilities(offline)
    assert np.allclose([seen["probs"][c] for c in bundle.classes], expected, atol=1e-12)


def test_gru_numpy_forward_basics():
    H, I, T = 4, 3, 5
    zeros = dict(w_ih=np.zeros((3 * H, I)), w_hh=np.zeros((3 * H, H)), b_ih=np.zeros(3 * H), b_hh=np.zeros(3 * H),
                 w_out=np.eye(3, H), b_out=np.array([1., 2., 3.]))
    x = np.random.default_rng(0).normal(size=(T, I))
    assert np.allclose(gru_logits(zeros, x, np.ones(T)), [1, 2, 3])                      # h stays 0
    w = dict(zeros, w_ih=np.random.default_rng(1).normal(size=(3 * H, I)), w_hh=np.random.default_rng(2).normal(size=(3 * H, H)))
    full = gru_logits(w, x, np.array([1, 1, 0, 0, 0.]))
    assert np.allclose(full, gru_logits(w, x[:2], np.ones(2)))                           # masked steps carry state
    assert np.isfinite(gru_logits(w, x * 1e3, np.ones(T))).all()
