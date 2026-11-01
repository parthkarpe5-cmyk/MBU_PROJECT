"""SYNTHETIC data generators. For plumbing/tests/smoke ONLY -- never evidence about drowsiness.

(1) ``write_synthetic_videos``: renders short videos from a public-domain still (scikit-image's bundled
    photo) with sensor-like noise/jitter, so the REAL MediaPipe extraction path can be exercised.
(2) ``write_synthetic_cache``: fabricates per-frame records with class-dependent statistics, so training/evaluation
    mechanics can be exercised quickly. Any metric computed from it is meaningless by construction.
"""
import json
from pathlib import Path

import numpy as np

from dms.features import FEATURE_VERSION
from .config import dump_json

LABELS = {"alert": 0, "low_vigilance": 5, "drowsy": 10}
SMOKE_LAYOUT_REGEX = r"Fold(?P<fold>\d+)/(?P<participant>[^/]+)/(?P<stem>\d+)\.mp4"


def write_synthetic_videos(root, folds=5, per_fold=2, seconds=12, fps=10, seed=0):
    import cv2
    from skimage import data
    rng = np.random.default_rng(seed)
    still = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)[0:300, 100:420]
    base = cv2.resize(still, (640, 480))
    count = 0
    for fold in range(1, folds + 1):
        for k in range(per_fold):
            participant = f"p{(fold - 1) * per_fold + k + 1:02d}"
            for name, label in LABELS.items():
                path = Path(root) / f"Fold{fold}" / participant / f"{label}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
                if not writer.isOpened():
                    raise RuntimeError("cv2.VideoWriter could not open an mp4v writer in this environment")
                for _ in range(int(seconds * fps)):
                    dx, dy = rng.integers(-2, 3, size=2)
                    frame = np.roll(base, (int(dy), int(dx)), axis=(0, 1)).astype(np.int16)
                    frame = np.clip(frame + rng.normal(0, 3, frame.shape), 0, 255).astype(np.uint8)
                    writer.write(frame)
                writer.release()
                count += 1
    return count


def write_synthetic_cache(cache_dir, key="synthetic", folds=5, per_fold=3, seconds=150, fps=15.0, seed=0, class_signal=1.0):
    """Fabricate caches + manifest rows. Class d in {0, .5, 1}: more/longer closures, more mouth opening for higher d."""
    rng = np.random.default_rng(seed)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(1, folds + 1):
        split = "train" if fold <= 3 else "val" if fold == 4 else "test"
        for k in range(per_fold):
            pid = f"P{(fold - 1) * per_fold + k + 1}"
            base = rng.uniform(.26, .34)
            for (name, label), d in zip(LABELS.items(), (0.0, 0.5, 1.0)):
                d *= class_signal
                n = int(seconds * fps)
                t = np.arange(n) / fps
                ear = base + rng.normal(0, .008, n)
                closed = np.zeros(n, bool)
                for s in rng.uniform(0, seconds, int((0.25 + 0.15 * d) * seconds / 4)):
                    dur = rng.uniform(.1, .25) + d * rng.uniform(0, .6)
                    closed[(t >= s) & (t < s + dur)] = True
                ear = np.where(closed, base * .35 + rng.normal(0, .01, n), ear)
                mar = .12 + rng.normal(0, .01, n)
                for s in rng.uniform(0, seconds, int(1 + 3 * d)):
                    mar[(t >= s) & (t < s + 2.0)] += .5
                yaw = np.cumsum(rng.normal(0, .15 * (1 + d), n)) * .1
                pitch = 5 + np.cumsum(rng.normal(0, .15 * (1 + d), n)) * .1
                roll = rng.normal(0, 1, n)
                valid = np.ones(n, bool)
                for s in rng.uniform(0, seconds, 4):          # realistic tracking dropouts: MISSING, not closed
                    valid[(t >= s) & (t < s + rng.uniform(.3, 2.0))] = False
                detected = valid.copy()
                arrays = dict(t=t, detected=detected, valid=valid, ear_l=np.where(valid, ear, np.nan), ear_r=np.where(valid, ear + rng.normal(0, .004, n), np.nan),
                              mar=np.where(valid, mar, np.nan), yaw=np.where(valid, yaw, np.nan), pitch=np.where(valid, pitch, np.nan),
                              roll=np.where(valid, roll, np.nan))
                vid = f"{pid}__L{label}"
                np.savez_compressed(cache_dir / f"{vid}.npz", **arrays)
                dump_json(cache_dir / f"{vid}.json", {"extraction_key": key, "synthetic": True, "feature_version": FEATURE_VERSION, "duration_s": float(t[-1])})
                rows.append(dict(participant_id=pid, fold=fold, video_id=vid, video_path=f"synthetic/{vid}.mp4", label=label, label_name=name,
                                 split=split, duration_s=seconds, fps=fps))
    return rows
