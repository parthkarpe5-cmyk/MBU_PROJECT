"""Configuration, seeding, hashing and environment capture."""
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "seed": 1337,
    "dataset": {
        "name": "UTA-RLDD",
        "root": None,                 # or env RLDD_ROOT or --dataset-root. Raw videos stay private/out of git.
        "layout_regex": None,         # REQUIRED: set after `inspect` shows the real layout (named groups: fold, participant)
        "video_extensions": [".mp4", ".mov", ".avi", ".mkv", ".m4v"],
        # Class names for the dataset's documented numeric labels (source below). Nothing else is assumed.
        "label_map": {"0": "alert", "5": "low_vigilance", "10": "drowsy"},
        "label_map_provenance": "UTA-RLDD homepage https://sites.google.com/view/utarldd/home : "
                                "'labeled as belonging to one of three classes: alert (labeled as 0), "
                                "low vigilant (labeled as 5) and drowsy (labeled as 10)'",
        # label_source: {"type": "csv", "path": "...", "provenance": "..."}   (columns: video_path,label)
        #            or {"type": "filename_stem", "documentation": "<where the dataset documents this>",
        #                "confirmed": false}   (requires confirmed=true; every stem must be in label_map)
        "label_source": None,
        "expected": {"participants": 60, "videos": 180, "folds": 5, "participants_per_fold": 12},
    },
    "splits": {"train_folds": [1, 2, 3], "val_folds": [4], "test_folds": [5]},
    "extraction": {"target_fps": 15.0, "max_side": 640, "timestamp_source": "frame_index",
                   "face_model": "models/face_landmarker.task"},
    "features": {},                   # overrides for dms.features.FeatureConfig
    "baseline": {"window_seconds_grid": [30.0, 60.0], "C_grid": [0.01, 0.1, 1.0, 10.0]},
    "gru": {"window_seconds": 30.0, "hidden_grid": [16, 32], "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4,
            "batch_size": 128, "max_epochs": 40, "patience": 6, "device": "auto"},
    "evaluation": {"bootstrap_samples": 1000, "latency_windows": 300},
}


def deep_merge(base, extra):
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = deep_merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    cfg = deep_merge(cfg, overrides or {})
    splits = cfg["splits"]
    seen = {}
    for name in ("train_folds", "val_folds", "test_folds"):
        for fold in splits[name]:
            if fold in seen:
                raise ValueError(f"Fold {fold} is assigned to both {seen[fold]} and {name}")
            seen[fold] = name
    return cfg


def feature_config(cfg, **overrides):
    from dms.features import FeatureConfig
    return FeatureConfig.from_dict({**cfg["features"], **overrides})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "numpy": np.__version__}
    for name in ("cv2", "mediapipe", "sklearn", "torch", "scipy"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "?")
        except Exception:
            info[name] = None
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        info["cuda_available"] = None
    return info


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)
