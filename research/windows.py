"""Cached per-frame records -> labelled windows (weak supervision: window label = its video's label).

Windows are generated per video on a fixed grid, never across videos, and never cross splits because
every video inherits the split of its participant's fold. Windows overlap when stride < window; overlapping
windows are only ever compared within the same participant/split.
"""
import json

import numpy as np

from dms.features import (MODEL_FEATURE_NAMES, SEQUENCE_CHANNELS, feature_vector, iter_window_bounds,
                          sequence_bins, window_features)
from .dataset import LeakageError, assert_participant_disjoint
from .extract import load_cache

CLASS_NAMES = ("alert", "low_vigilance", "drowsy")


def class_index(label_name):
    return CLASS_NAMES.index(label_name)


def video_end_time(arrays):
    t = arrays["t"]
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    return float(t[-1] + step)


def build_windows(rows, cache_dir, fcfg, kind="tabular", splits=None, log=print):
    """Return dict(meta=list[dict], X=..., y=..., stats=...). ``splits`` limits which splits are even loaded
    (training passes ('train','val'); the test split is never touched by training code)."""
    selected = [r for r in rows if splits is None or r["split"] in splits]
    assert_participant_disjoint(selected)
    meta, feats, seqs, masks = [], [], [], []
    stats = {"windows_total": 0, "windows_kept": 0, "dropped_low_coverage": 0, "videos": 0, "videos_missing_cache": []}
    for r in selected:
        try:
            arrays, cache_meta = load_cache(cache_dir, r["video_id"])
        except FileNotFoundError:
            stats["videos_missing_cache"].append(r["video_id"])
            continue
        if cache_meta.get('max_seconds') is not None or cache_meta.get('possibly_truncated'):
            raise ValueError(f"Incomplete extraction cache for {r['video_id']}; run full extraction before training.")
        stats["videos"] += 1
        t = arrays["t"]
        if len(t) < 2:
            continue
        for t0, t1 in iter_window_bounds(float(t[0]), video_end_time(arrays), fcfg):
            stats["windows_total"] += 1
            f = window_features(arrays, t0, t1, fcfg)
            if f["coverage"] < fcfg.min_coverage or f["n_valid"] < fcfg.min_valid_samples:
                stats["dropped_low_coverage"] += 1
                continue
            if kind == "sequence":
                values, mask = sequence_bins(arrays, t0, t1, fcfg)
                if mask.sum() < 1:
                    stats["dropped_low_coverage"] += 1
                    continue
                seqs.append(values)
                masks.append(mask)
            feats.append(feature_vector(f))
            meta.append(dict(participant_id=r["participant_id"], video_id=r["video_id"], split=r["split"], fold=r["fold"],
                             y=class_index(r["label_name"]), t0=t0, t1=t1, coverage=f["coverage"]))
            stats["windows_kept"] += 1
    assert_participant_disjoint(meta)
    out = dict(meta=meta, stats=stats, y=np.array([m["y"] for m in meta], dtype=int))
    out["X"] = np.array(feats) if feats else np.zeros((0, len(MODEL_FEATURE_NAMES)))
    if kind == "sequence":
        out["V"] = np.array(seqs) if seqs else np.zeros((0, fcfg.n_bins, len(SEQUENCE_CHANNELS)))
        out["M"] = np.array(masks) if masks else np.zeros((0, fcfg.n_bins))
    if stats["videos_missing_cache"]:
        log(f"WARNING: {len(stats['videos_missing_cache'])} videos have no cache and were skipped: {stats['videos_missing_cache'][:5]}")
    return out


def subset(windows, split):
    idx = [i for i, m in enumerate(windows["meta"]) if m["split"] == split]
    out = {"meta": [windows["meta"][i] for i in idx], "y": windows["y"][idx], "X": windows["X"][idx]}
    if "V" in windows:
        out["V"], out["M"] = windows["V"][idx], windows["M"][idx]
    return out


def summarize_windows(windows):
    by_class = {}
    for name_idx, name in enumerate(CLASS_NAMES):
        by_class[name] = int((windows["y"] == name_idx).sum())
    return {"n_windows": int(len(windows["y"])), "n_participants": len({m["participant_id"] for m in windows["meta"]}),
            "n_videos": len({m["video_id"] for m in windows["meta"]}), "windows_by_class": by_class}
