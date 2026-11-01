"""Training/evaluation mechanics on SYNTHETIC caches (no claim about real accuracy), plus real-MediaPipe parity."""
import json
from pathlib import Path
import numpy as np
import pytest
import research.windows as windows_mod
from dms.learned import Bundle
from research.baseline import train_baseline
from research.config import load_config
from research.evaluate import evaluate_run
from research.export import export_bundle
from research.heuristic import evaluate_heuristic, replay
from research.metrics import auroc, cluster_bootstrap, confusion_matrix, ece, fit_temperature, macro_f1, summarize
from research.synthetic import write_synthetic_cache

CFG = load_config(None, {"features": {"window_seconds": 30, "stride_seconds": 10}, "baseline": {"window_seconds_grid": [30.0], "C_grid": [0.1, 1.0]},
                         "evaluation": {"bootstrap_samples": 50, "latency_windows": 20}})


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    rows = write_synthetic_cache(root / "cache", folds=5, per_fold=3, seconds=150)
    return root, rows


def test_training_never_loads_test_videos(synthetic, monkeypatch):
    root, rows = synthetic
    loaded = []
    original = windows_mod.load_cache
    monkeypatch.setattr(windows_mod, "load_cache", lambda d, v: (loaded.append(v), original(d, v))[1])
    train_baseline(CFG, rows, root / "cache", root / "run", "sha", synthetic=True, log=lambda m: None)
    test_ids = {r["video_id"] for r in rows if r["split"] == "test"}
    assert loaded and not (set(loaded) & test_ids)


def test_baseline_bundle_and_evaluation_are_consistent(synthetic):
    root, rows = synthetic
    train_baseline(CFG, rows, root / "cache", root / "run2", "sha", synthetic=True, log=lambda m: None)
    meta = json.loads((root / "run2" / "bundle.json").read_text())
    assert meta["training"]["test_split_used"] is False and meta["training"]["synthetic"] is True
    assert meta["training"]["preprocessing_fit_split"] == "train" and meta["training"]["calibration_fit_split"] == "validation"
    result = evaluate_run(root / "run2", rows, root / "cache", CFG, "test", log=lambda m: None)
    assert result["windows"]["participants"] == 3
    for key in ("macro_f1", "balanced_accuracy", "per_class", "confusion_matrix"):
        assert key in result["window_level"]
    assert "per_participant" in result["participant_variability"] and result["latency"]["per_frame_predictor_update"]["n"] > 0
    assert (root / "run2" / "report_test.md").is_file()
    assert Bundle(root / "run2").meta["validation_status"]["heldout_evaluated"] is True
    again = evaluate_run(root / "run2", rows, root / "cache", CFG, "test", log=lambda m: None)
    assert again["previous_evaluations_of_this_split"] == 1                      # repeated test looks are recorded


def test_synthetic_bundle_cannot_be_exported_to_app_by_accident(synthetic, tmp_path):
    root, rows = synthetic
    train_baseline(CFG, rows, root / "cache", root / "run3", "sha", synthetic=True, log=lambda m: None)
    with pytest.raises(RuntimeError, match="SYNTHETIC"):
        export_bundle(root / "run3", tmp_path / "dest")
    with pytest.raises(RuntimeError, match="Held-out"):
        export_bundle(root / "run3", tmp_path / "dest", allow_synthetic=True, require_heldout=True)
    export_bundle(root / "run3", tmp_path / "dest", allow_synthetic=True)
    assert Bundle(tmp_path / "dest").model_type == "logreg"


def test_heuristic_replay_matches_live_monitor_semantics(synthetic):
    root, rows = synthetic
    report = evaluate_heuristic(rows, root / "cache", CFG, root / "hx", log=lambda m: None)
    assert report["primary_measure"] == "hx_any_alert_frac" and "no like-for-like" in report["caveat"]
    from research.extract import load_cache
    arrays, _ = load_cache(root / "cache", rows[0]["video_id"])
    flags = replay(arrays)
    assert len(flags["eyes"]) == len(arrays["t"])
    assert not flags["eyes"][~arrays["valid"]].any()                        # invalid frames never raise alerts


def test_metrics_basics():
    y, p = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 0])
    s = summarize(y, p)
    assert s["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert s["macro_f1"] == pytest.approx(macro_f1(y, p)) and s["per_class"]["alert"]["support"] == 2
    assert auroc(np.array([0, 1, 2, 3.]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert auroc(np.array([1, 1, 1, 1.]), np.array([0, 0, 1, 1], bool)) == .5
    probs = np.array([[.9, .1, 0], [.1, .8, .1]])
    assert 0 <= ece(probs, np.array([0, 1])) <= 1
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 3)) * 5
    y = np.argmax(logits + rng.normal(size=(500, 3)) * 8, 1)
    assert fit_temperature(logits, y) > 1                                     # overconfident logits get cooled
    ci = cluster_bootstrap(y, logits.argmax(1), np.repeat(np.arange(50), 10), macro_f1, 100, 0)
    assert ci["low"] <= ci["high"]


# ---- real MediaPipe parity ---------------------------------------------------------------
def test_extractor_matches_live_frame_path(tmp_path):
    cv2 = pytest.importorskip("cv2")
    model = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
    if not model.exists():
        pytest.skip("needs models/face_landmarker.task")
    from research.synthetic import write_synthetic_videos
    from research.extract import extract_video, extraction_key
    from dms.features import frame_record
    from dms.vision import Tracker
    write_synthetic_videos(tmp_path, folds=1, per_fold=1, seconds=2, fps=10)
    video = tmp_path / "Fold1" / "p01" / "0.mp4"
    cfg = load_config(None, {"extraction": {"target_fps": 10.0}})
    tracker = Tracker(model)
    try:
        meta = extract_video(video, tmp_path / "v.npz", cfg, tracker, "k")
    finally:
        tracker.close()
    with np.load(tmp_path / "v.npz") as cached:
        cache = {k: cached[k] for k in cached.files}
    # Re-run the live path on the identical decoded frames with a fresh tracker.
    tracker = Tracker(model)
    try:
        cap, i, checked = cv2.VideoCapture(str(video)), 0, 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            if i < len(cache["t"]) and abs(cache["t"][i] - t) < 1e-9:
                live = frame_record(tracker.process(frame, t))
                assert live["valid"] == bool(cache["valid"][i])
                for k in ("ear_l", "ear_r", "mar", "yaw", "pitch", "roll"):
                    assert np.allclose(live[k], cache[k][i], equal_nan=True, atol=1e-9)
                checked += 1
            i += 1
        cap.release()
    finally:
        tracker.close()
    assert checked >= 15 and meta["frames_processed"] == len(cache["t"])


def test_truncated_smoke_cache_is_not_reused_for_full_run(tmp_path):
    from research.extract import cache_paths, is_cached
    npz, js = cache_paths(tmp_path, "v")
    npz.write_bytes(b"x")
    js.write_text(json.dumps({"extraction_key": "k", "max_seconds": 60}))
    assert is_cached(tmp_path, "v", "k", 60) and not is_cached(tmp_path, "v", "k", None)
    js.write_text(json.dumps({"extraction_key": "k", "max_seconds": None}))
    assert is_cached(tmp_path, "v", "k", None) and not is_cached(tmp_path, "v", "other")
