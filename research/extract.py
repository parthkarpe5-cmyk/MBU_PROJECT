"""Video -> timestamped per-frame observation records (cached, resumable).

Uses the SAME ``dms.vision.Tracker`` (MediaPipe Face Landmarker, EAR/MAR, solvePnP pose, `healthy`
check) and the SAME ``dms.features.frame_record`` as the live app. Differences from live capture,
all recorded in the cache metadata and discussed in MODEL_CARD.md:
  * frames are decoded from compressed video, not a 640x480 webcam stream;
  * frames larger than ``max_side`` are downscaled with aspect ratio preserved (no distortion);
  * frames are sampled at ``target_fps`` by video timestamp.
"""
import json
import os
from pathlib import Path
import time

import numpy as np

from dms.features import FEATURE_VERSION, FRAME_FIELDS, frame_record
from .config import ROOT, dump_json, sha256_file, sha256_json

EXTRACTOR_VERSION = "1"


def live_definition_hash():
    """Hash of the source files that define live measurements; changing them invalidates caches."""
    parts = [(ROOT / "dms" / name).read_bytes() for name in ("core.py", "vision.py", "features.py")]
    return sha256_json([p.hex() for p in parts])[:16]


def extraction_key(cfg, face_model_path):
    ex = cfg["extraction"]
    try:
        import mediapipe
        mp_version = mediapipe.__version__
    except Exception:
        mp_version = None
    return sha256_json({
        "extractor": EXTRACTOR_VERSION, "feature_version": FEATURE_VERSION, "target_fps": ex["target_fps"],
        "max_side": ex["max_side"], "timestamp_source": ex["timestamp_source"],
        "face_model_sha256": sha256_file(face_model_path), "mediapipe": mp_version, "live_definitions": live_definition_hash(),
    })[:16]


def prepare_frame(frame, max_side):
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    import cv2
    scale = max_side / max(h, w)
    return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def observe(tracker, frame, t, max_side):
    """One frame -> observation record. The single code path used for every dataset frame."""
    return frame_record(tracker.process(prepare_frame(frame, max_side), t))


def extract_video(video_path, out_path, cfg, tracker, key, max_seconds=None):
    import cv2
    ex = cfg["extraction"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    started = time.perf_counter()
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        use_index = ex["timestamp_source"] == "frame_index" and fps and np.isfinite(fps) and 1 <= fps <= 240
        interval = 1.0 / ex["target_fps"]
        next_t, index, read = 0.0, 0, 0
        ts, records = [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read += 1
            t = index / fps if use_index else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            index += 1
            if max_seconds is not None and t > max_seconds:
                break
            if t + 1e-9 < next_t:
                continue
            while next_t <= t + 1e-9:
                next_t += interval
            ts.append(t)
            records.append(observe(tracker, frame, t, ex["max_side"]))
    finally:
        cap.release()
    if not ts:
        raise RuntimeError(f"No frames decoded from {video_path}")
    t = np.asarray(ts, dtype=np.float64)
    if np.any(np.diff(t) < 0):
        raise RuntimeError(f"Non-monotonic timestamps in {video_path}")
    arrays = {"t": t, "detected": np.array([r["detected"] for r in records]), "valid": np.array([r["valid"] for r in records])}
    for name in FRAME_FIELDS:
        arrays[name] = np.array([r[name] for r in records], dtype=np.float64)
    meta = {
        "video": Path(video_path).name, "extraction_key": key, "video_fps": float(fps) if fps else None,
        "timestamp_source": "frame_index" if use_index else "pos_msec", "frames_decoded": read,
        "frames_processed": len(t), "expected_frames": int(expected_frames) if expected_frames and expected_frames > 0 else None,
        "possibly_truncated": bool(expected_frames and expected_frames > 0 and read < 0.98 * expected_frames and max_seconds is None),
        "duration_s": float(t[-1] - t[0]), "valid_fraction_frames": float(arrays["valid"].mean()),
        "detected_fraction_frames": float(arrays["detected"].mean()), "elapsed_s": round(time.perf_counter() - started, 2),
        "target_fps": ex["target_fps"], "max_side": ex["max_side"], "feature_version": FEATURE_VERSION,
        "max_seconds": max_seconds,   # non-null = deliberately truncated (smoke) cache
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(tmp, out_path)
    dump_json(out_path.with_suffix(".json"), meta)
    return meta


def cache_paths(cache_dir, video_id):
    base = Path(cache_dir) / video_id
    return base.with_suffix(".npz"), base.with_suffix(".json")


def is_cached(cache_dir, video_id, key, max_seconds=None):
    npz, js = cache_paths(cache_dir, video_id)
    if not (npz.is_file() and js.is_file()):
        return False
    try:
        meta = json.loads(js.read_text(encoding="utf-8"))
        return meta.get("extraction_key") == key and meta.get("max_seconds") == max_seconds
    except ValueError:
        return False


def load_cache(cache_dir, video_id):
    npz, js = cache_paths(cache_dir, video_id)
    with np.load(npz, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return arrays, json.loads(js.read_text(encoding="utf-8"))


# ---- worker-process plumbing -------------------------------------------------------------
_TRACKER = None


def _init_worker(model_path):
    global _TRACKER
    from dms.vision import Tracker
    _TRACKER = Tracker(model_path)


def _work(job):
    video, out, cfg, key, max_seconds = job
    try:
        return ("ok", out, extract_video(video, out, cfg, _TRACKER, key, max_seconds))
    except Exception as exc:
        return ("fail", out, f"{type(exc).__name__}: {exc}")


def extract_all(rows, root, cache_dir, cfg, workers=1, force=False, max_videos=None, max_seconds=None, log=print):
    face_model = Path(cfg["extraction"]["face_model"])
    if not face_model.is_absolute():
        face_model = ROOT / face_model
    if not face_model.is_file():
        raise FileNotFoundError(f"Face model missing: {face_model} (run setup_model.py)")
    key = extraction_key(cfg, face_model)
    cache_dir = Path(cache_dir) / key
    jobs, skipped = [], 0
    for r in rows[: max_videos or None]:
        if not force and is_cached(cache_dir, r["video_id"], key, max_seconds):
            skipped += 1
            continue
        out, _ = cache_paths(cache_dir, r["video_id"])
        jobs.append((str(Path(root) / r["video_path"]), str(out), cfg, key, max_seconds))
    log(f"extraction key {key}: {len(jobs)} to process, {skipped} already cached")
    failures, done = [], 0
    started = time.perf_counter()
    if workers <= 1:
        _init_worker(str(face_model))
        results = map(_work, jobs)
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                      initializer=_init_worker, initargs=(str(face_model),))
        results = pool.map(_work, jobs)
    try:
        for status, out, info in results:
            done += 1
            if status == "ok":
                log(f"[{done}/{len(jobs)}] {Path(out).name}: {info['frames_processed']} frames, valid {info['valid_fraction_frames']:.0%}, "
                    f"{info['elapsed_s']}s" + (" POSSIBLY TRUNCATED" if info["possibly_truncated"] else ""))
            else:
                failures.append((out, info))
                log(f"[{done}/{len(jobs)}] FAILED {Path(out).name}: {info}")
    finally:
        if workers > 1:
            pool.shutdown()
    if failures:
        dump_json(cache_dir / "extract_failures.json", [{"output": o, "error": e} for o, e in failures])
    log(f"done in {time.perf_counter() - started:.1f}s; failures: {len(failures)}")
    return key, cache_dir, failures
