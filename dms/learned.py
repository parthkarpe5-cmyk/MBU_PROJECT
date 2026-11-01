"""Optional local learned driver-state model (research prototype).

Contract (see MODEL_CARD.md for the prose version)
  Input : rolling window of per-frame observation records (``dms.features``), RAW uncalibrated
          tracker values at the live frame rate. Window length/stride/thresholds come from the
          bundle's ``feature_config`` so training and live definitions cannot drift.
  Output: class scores over ("alert", "low_vigilance", "drowsy") for the LAST completed window,
          its window-end time (-> prediction age), and tracking coverage.
  Not   : facial expression, event detection, or an alarm. It never changes heuristic alerts.

Runtime is numpy-only (no torch/sklearn) and fully offline. Any problem -- missing bundle,
hash mismatch, version mismatch, exception during inference -- yields a status dict with
``state`` "disabled" or "error"; it never raises into the camera loop.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .features import (FEATURE_VERSION, FeatureConfig, MODEL_FEATURE_NAMES, SEQUENCE_CHANNELS,
                       SequencePreprocessor, TabularPreprocessor, feature_vector, row_from_record,
                       rows_to_arrays, sequence_bins, window_coverage, window_features)

SCHEMA_VERSION = 1
CLASSES = ("alert", "low_vigilance", "drowsy")
BUNDLE_FILE, WEIGHTS_FILE = "bundle.json", "weights.npz"


class BundleError(Exception):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def gru_logits(w, x, mask):
    """Single-sequence GRU forward, identical to torch.nn.GRUCell (gate order r, z, n) with
    masked steps skipped (hidden state carried over). x: [T, I]; mask: [T]."""
    hidden = w["w_hh"].shape[1]
    h = np.zeros(hidden)
    for step in range(len(x)):
        if mask[step] < 0.5:
            continue
        gi = w["w_ih"] @ x[step] + w["b_ih"]
        gh = w["w_hh"] @ h + w["b_hh"]
        r = sigmoid(gi[:hidden] + gh[:hidden])
        z = sigmoid(gi[hidden:2 * hidden] + gh[hidden:2 * hidden])
        n = np.tanh(gi[2 * hidden:] + r * gh[2 * hidden:])
        h = (1.0 - z) * n + z * h
    return w["w_out"] @ h + w["b_out"]


class Bundle:
    def __init__(self, directory):
        directory = Path(directory)
        meta_path, weights_path = directory / BUNDLE_FILE, directory / WEIGHTS_FILE
        if not meta_path.is_file() or not weights_path.is_file():
            raise BundleError(f"No trained bundle in {directory} (need {BUNDLE_FILE} and {WEIGHTS_FILE})")
        try:
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise BundleError(f"Unreadable {BUNDLE_FILE}: {exc}") from exc
        meta = self.meta
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise BundleError(f"Unsupported bundle schema {meta.get('schema_version')!r}")
        if meta.get("feature_version") != FEATURE_VERSION:
            raise BundleError(f"Bundle feature version {meta.get('feature_version')!r} != runtime {FEATURE_VERSION!r}; "
                              "feature definitions changed, retrain/re-export")
        if tuple(meta.get("classes", ())) != CLASSES:
            raise BundleError(f"Unexpected classes {meta.get('classes')!r}")
        if sha256_file(weights_path) != meta.get("weights_sha256"):
            raise BundleError("weights.npz does not match the hash recorded in bundle.json")
        try:
            self.cfg = FeatureConfig.from_dict(meta["feature_config"])
            self.model_type = meta["model_type"]
            self.classes = CLASSES
            with np.load(weights_path, allow_pickle=False) as npz:
                self.weights = {k: np.asarray(npz[k], dtype=np.float64) for k in npz.files}
            cal = meta.get("calibration") or {}
            self.temperature = float(cal.get("temperature", 1.0)) if cal.get("method") == "temperature" else 1.0
            if not np.isfinite(self.temperature) or self.temperature <= 0:
                raise BundleError("Invalid calibration temperature")
            if self.model_type == "logreg":
                self.pre = TabularPreprocessor.from_dict(meta["preprocessing"])
                if tuple(self.pre.names) != MODEL_FEATURE_NAMES:
                    raise BundleError("Bundle feature names differ from runtime MODEL_FEATURE_NAMES")
                if self.weights["W"].shape != (3, self.pre.output_dim) or self.weights["b"].shape != (3,):
                    raise BundleError("logreg weight shapes do not match preprocessing")
            elif self.model_type == "gru":
                self.pre = SequencePreprocessor.from_dict(meta["preprocessing"])
                if tuple(self.pre.channels) != SEQUENCE_CHANNELS:
                    raise BundleError("Bundle sequence channels differ from runtime SEQUENCE_CHANNELS")
                hidden = self.weights["w_hh"].shape[1]
                if self.weights["w_ih"].shape != (3 * hidden, self.pre.input_dim) or self.weights["w_out"].shape != (3, hidden):
                    raise BundleError("gru weight shapes do not match preprocessing")
            else:
                raise BundleError(f"Unknown model_type {self.model_type!r}")
        except KeyError as exc:
            raise BundleError(f"Bundle is missing {exc}") from exc

    # -- scoring
    def logits_from_window(self, arrays, t0, t1):
        """Deployed scoring path for [t0, t1). Returns (logits, window_features) or (None, features)
        when the window lacks enough valid data."""
        cfg = self.cfg
        feats = window_features(arrays, t0, t1, cfg)
        if feats["coverage"] < cfg.min_coverage or feats["n_valid"] < cfg.min_valid_samples:
            return None, feats
        if self.model_type == "logreg":
            x = self.pre.transform(feature_vector(feats)[None, :])[0]
            return self.weights["W"] @ x + self.weights["b"], feats
        values, mask = sequence_bins(arrays, t0, t1, cfg)
        if mask.sum() < 1:
            return None, feats
        return gru_logits(self.weights, self.pre.transform(values, mask), mask), feats

    def probabilities(self, logits):
        return softmax(logits, self.temperature)

    @property
    def score_kind(self):
        cal = self.meta.get("calibration") or {}
        if cal.get("method") not in ("temperature",):
            return "uncalibrated"
        held = (self.meta.get("validation_status") or {}).get("heldout_calibration")
        return "calibrated_heldout_checked" if held else "calibrated_unverified"

    def score_description(self, probability):
        kind = self.score_kind
        if kind == "uncalibrated":
            return f"model score {probability:.2f} (UNCALIBRATED softmax; not a probability)"
        if kind == "calibrated_unverified":
            return f"calibrated score {probability:.2f} (fit on validation windows; held-out check NOT recorded)"
        held = self.meta["validation_status"]["heldout_calibration"]
        return (f"calibrated probability {probability:.2f} (held-out ECE {held.get('ece', float('nan')):.2f} on "
                f"{held.get('n_participants', '?')} participants; agreement with weak video labels only)")


class LearnedStatePredictor:
    """Rolling-window inference for the live camera loop. Not thread-safe: call from one thread."""

    def __init__(self, bundle=None, error=None, max_age_seconds=None, max_valid_age_seconds=2.0, enabled=True):
        self.bundle, self.error, self.enabled = bundle, error, enabled
        self.rows = deque()
        self.first_t = self.last_t = self.last_valid_t = self.last_end = None
        self.prediction = None
        self.max_valid_age = max_valid_age_seconds
        cfg = bundle.cfg if bundle else FeatureConfig()
        self.max_age = max_age_seconds if max_age_seconds is not None else 2 * cfg.stride_seconds + 1

    @classmethod
    def from_path(cls, path, **options):
        """Never raises. A missing/invalid bundle yields a disabled predictor with a reason."""
        if not path:
            return cls(error="No trained model configured", **options)
        try:
            return cls(Bundle(path), **options)
        except BundleError as exc:
            return cls(error=str(exc), **options)
        except Exception as exc:  # corrupted npz, etc.
            return cls(error=f"{type(exc).__name__}: {exc}", **options)

    def reset(self):
        self.rows.clear()
        self.first_t = self.last_t = self.last_valid_t = self.last_end = None
        self.prediction = None

    def _status(self, state, message, coverage=None):
        prediction = self.prediction if state == "ok" else None
        score_text = None
        if prediction is not None:
            score_text = self.bundle.score_description(prediction["probs"][prediction["label"]])
        return dict(state=state, message=message, prediction=prediction, coverage=coverage, score_text=score_text,
                    score_kind=self.bundle.score_kind if self.bundle else None,
                    window_seconds=self.bundle.cfg.window_seconds if self.bundle else None)

    def update(self, t, record):
        """Feed one observation (``record`` from ``dms.features.frame_record``). Returns status dict."""
        if self.bundle is None:
            return self._status("disabled", self.error or "No trained model")
        if not self.enabled:
            return self._status("disabled", "Learned model switched off")
        try:
            return self._update(float(t), record)
        except Exception as exc:  # inference must never take down the heuristic monitor
            self.error = f"{type(exc).__name__}: {exc}"
            self.bundle = None
            self.prediction = None
            return self._status("error", f"Inference failed, heuristics unaffected: {self.error}")

    def _update(self, t, record):
        cfg = self.bundle.cfg
        if self.last_t is not None and t < self.last_t:
            self.reset()                       # clock went backwards: do not stitch windows
        row = row_from_record(t, record)
        if self.first_t is None:
            self.first_t = t
        self.last_t = t
        if row[1]:
            self.last_valid_t = t
        t1, t0 = t, t - cfg.window_seconds
        self.rows.append(row)
        while self.rows and self.rows[0][0] < t0:
            self.rows.popleft()
        if t - self.first_t < cfg.window_seconds:
            return self._status("warming", f"Collecting first window: {t - self.first_t:.0f}/{cfg.window_seconds:.0f} s")
        arrays = rows_to_arrays(self.rows)
        coverage, _, n_valid = window_coverage(arrays, t0, t1, cfg)
        if self.last_valid_t is None or t - self.last_valid_t > self.max_valid_age:
            return self._status("insufficient", "No valid face observation recently; prediction suppressed", coverage)
        if coverage < cfg.min_coverage or n_valid < cfg.min_valid_samples:
            return self._status("insufficient", f"Tracking coverage {coverage:.0%} < {cfg.min_coverage:.0%}; prediction suppressed", coverage)
        if self.last_end is None or t - self.last_end >= cfg.stride_seconds:
            started = time.perf_counter()
            logits, feats = self.bundle.logits_from_window(arrays, t0, t1)
            if logits is None:
                return self._status("insufficient", "Not enough valid observations in window", coverage)
            probs = self.bundle.probabilities(logits)
            if not np.isfinite(probs).all():
                raise ValueError("non-finite model output")
            self.prediction = dict(label=self.bundle.classes[int(np.argmax(probs))], probs=dict(zip(self.bundle.classes, map(float, probs))),
                                   window_end=t, coverage=coverage, latency_ms=(time.perf_counter() - started) * 1000)
            self.last_end = t
        if self.prediction is None:
            return self._status("warming", "Waiting for first prediction", coverage)
        if t - self.prediction["window_end"] > self.max_age:
            return self._status("stale", f"Last prediction is {t - self.prediction['window_end']:.0f} s old; suppressed", coverage)
        return self._status("ok", "", coverage)


def format_status(status, now=None):
    """Dashboard text for a predictor status (pure function; no Tk). ``now`` is on the same clock as
    ``update`` timestamps and is used to report prediction age."""
    if not status:
        return "Learned model: not initialised"
    head = "LEARNED MODEL (research; weak video-level training labels; never an alarm)\n"
    state = status["state"]
    if state in ("disabled", "error"):
        return head + f"Not active: {status['message']}\nHeuristic monitoring continues unchanged."
    if state in ("warming", "insufficient", "stale"):
        label = {"warming": "Warming up", "insufficient": "Insufficient data", "stale": "Stale"}[state]
        return head + f"{label}: {status['message']}\nNo learned class shown."
    prediction = status["prediction"]
    probs = prediction["probs"]
    score = status.get("score_text") or "model score (calibration status unknown; treat as UNCALIBRATED)"
    age = (now - prediction["window_end"]) if now is not None else float("nan")
    others = "  ".join(f"{k} {v:.2f}" for k, v in probs.items())
    return (head + f"Learned class: {prediction['label']}\n{score}\n{others}\n"
            f"Prediction age: {age:.1f} s | window {status['window_seconds']:.0f} s | tracking coverage {prediction['coverage']:.0%}")
