"""Write the runtime model bundle (bundle.json + weights.npz). Contract is read by dms.learned.Bundle."""
import datetime as dt
from pathlib import Path

import numpy as np

from dms.features import FEATURE_VERSION, MODEL_FEATURE_NAMES, SEQUENCE_CHANNELS
from dms.learned import BUNDLE_FILE, CLASSES, SCHEMA_VERSION, WEIGHTS_FILE, sha256_file
from .config import dump_json
from .metrics import ece, nll

INTENDED_USE = ("Research prototype. Window-level drowsiness-CLASS scores from facial-geometry features, trained on weak "
                "video-level labels. Not event detection, not facial-expression recognition, not an alarm, and not validated "
                "for automotive or safety use.")


def write_bundle(directory, model_type, weights, preprocessing, fcfg, val_logits, val_y, temperature, training, extra_status=None):
    from dms.learned import softmax
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / WEIGHTS_FILE, **{k: np.asarray(v, dtype=np.float64) for k, v in weights.items()})
    calibration = {"method": "none"}
    if temperature is not None and len(val_y):
        before, after = softmax(val_logits), softmax(val_logits, temperature)
        calibration = {"method": "temperature", "temperature": float(temperature), "fit_split": "validation",
                       "n_windows": int(len(val_y)), "val_ece_before": ece(before, val_y), "val_ece_after": ece(after, val_y),
                       "val_nll_before": nll(before, val_y), "val_nll_after": nll(after, val_y),
                       "note": "Fit against weak window labels (= video label). Measures agreement with those labels only."}
    meta = {
        "schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION, "model_type": model_type, "classes": list(CLASSES),
        "feature_config": fcfg.to_dict(), "preprocessing": preprocessing, "calibration": calibration,
        "input_contract": ({"kind": "window_features", "names": list(MODEL_FEATURE_NAMES)} if model_type == "logreg" else
                           {"kind": "sequence", "channels": list(SEQUENCE_CHANNELS), "n_bins": fcfg.n_bins, "bin_seconds": fcfg.bin_seconds,
                            "extra_channel": "observed-bin mask appended last"}),
        "output_contract": {"logits": "3 floats ordered as classes", "probabilities": "softmax(logits / temperature)"},
        "weights_sha256": sha256_file(directory / WEIGHTS_FILE),
        "training": training,
        "validation_status": {"heldout_evaluated": False, "heldout_calibration": None, "safety_validated": False, **(extra_status or {})},
        "intended_use": INTENDED_USE, "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    dump_json(directory / BUNDLE_FILE, meta)
    return meta
