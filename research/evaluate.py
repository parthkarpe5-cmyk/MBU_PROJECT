"""Held-out evaluation using the DEPLOYED scoring path (dms.learned.Bundle / LearnedStatePredictor).

All metrics are window-level agreement with weak video-level labels, aggregated per participant.
There are no event annotations, so no event-level accuracy is computed or claimed.
"""
import datetime as dt
import json
from pathlib import Path
import platform
import time

import numpy as np

from dms.features import FeatureConfig, iter_window_bounds, row_from_record
from dms.learned import Bundle, LearnedStatePredictor, softmax
from .config import dump_json, environment
from .extract import load_cache
from .metrics import (CLASS_NAMES, balanced_accuracy, cluster_bootstrap, ece, latency_stats, macro_f1, nll,
                      participant_breakdown, summarize)
from .windows import class_index, video_end_time


def score_split(bundle, rows, cache_dir, split, log=print):
    """Deployed scoring for every window of every video in `split`. Returns arrays + bookkeeping."""
    meta, logits, dropped, total = [], [], 0, 0
    cache = {}
    for r in rows:
        if r["split"] != split:
            continue
        try:
            arrays, cache_meta = load_cache(cache_dir, r["video_id"])
        except FileNotFoundError:
            log(f"WARNING: no cache for {r['video_id']}")
            continue
        if cache_meta.get('max_seconds') is not None or cache_meta.get('possibly_truncated'):
            raise ValueError(f"Incomplete extraction cache for {r['video_id']}; run full extraction before evaluation.")
        cache[r["video_id"]] = arrays
        if len(arrays["t"]) < 2:
            continue
        for t0, t1 in iter_window_bounds(float(arrays["t"][0]), video_end_time(arrays), bundle.cfg):
            total += 1
            z, _ = bundle.logits_from_window(arrays, t0, t1)
            if z is None:
                dropped += 1
                continue
            logits.append(z)
            meta.append(dict(participant_id=r["participant_id"], video_id=r["video_id"], y=class_index(r["label_name"]), t0=t0, t1=t1))
    return dict(meta=meta, logits=np.array(logits), total=total, dropped=dropped, cache=cache)


def measure_latency(bundle, scored, rows, n_windows, seed=0):
    rng = np.random.default_rng(seed)
    out = {"machine": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": environment()["cpu_count"]},
           "excludes": "camera capture and MediaPipe landmarking (measured separately by app FPS)"}
    if scored["meta"]:
        picks = rng.choice(len(scored["meta"]), size=min(n_windows, len(scored["meta"])), replace=False)
        times = []
        for i in picks:
            m = scored["meta"][i]
            started = time.perf_counter()
            bundle.logits_from_window(scored["cache"][m["video_id"]], m["t0"], m["t1"])
            times.append((time.perf_counter() - started) * 1000)
        out["window_scoring_features_plus_model"] = latency_stats(times)
    # Live-style replay: cost of predictor.update() on every processed frame, as in the camera loop.
    replay_times, vid = [], next(iter(scored["cache"]), None)
    if vid is not None:
        arrays = scored["cache"][vid]
        predictor = LearnedStatePredictor(bundle)
        for i in range(len(arrays["t"])):
            record = {"detected": True, "valid": bool(arrays["valid"][i]), **{k: float(arrays[k][i]) for k in ("ear_l", "ear_r", "mar", "yaw", "pitch", "roll")}}
            started = time.perf_counter()
            predictor.update(float(arrays["t"][i]), record)
            replay_times.append((time.perf_counter() - started) * 1000)
        out["per_frame_predictor_update"] = latency_stats(replay_times)
        out["per_frame_replay_video"] = vid
    return out


def evaluate_run(run_dir, rows, cache_dir, cfg, split="test", record_in_bundle=True, log=print):
    run_dir = Path(run_dir)
    bundle = Bundle(run_dir)
    log_path = run_dir / "eval_log.jsonl"
    previous = [json.loads(l) for l in log_path.read_text().splitlines()] if log_path.is_file() else []
    repeats = sum(1 for e in previous if e["split"] == split)
    if split == "test" and repeats:
        log(f"WARNING: this run's test split has already been evaluated {repeats} time(s); further looks risk test-set tuning.")
    scored = score_split(bundle, rows, cache_dir, split, log)
    if not len(scored["logits"]):
        raise RuntimeError(f"No scorable windows in split {split!r}")
    y = np.array([m["y"] for m in scored["meta"]])
    participants = np.array([m["participant_id"] for m in scored["meta"]])
    videos = np.array([m["video_id"] for m in scored["meta"]])
    raw_probs, probs = softmax(scored["logits"]), bundle.probabilities(scored["logits"])
    pred = probs.argmax(1)
    result = {
        "split": split, "evaluated": dt.datetime.now().isoformat(timespec="seconds"), "model_type": bundle.model_type,
        "bundle_weights_sha256": bundle.meta["weights_sha256"], "training": bundle.meta["training"],
        "label_semantics": "Window label = label of its whole video (weak supervision). No frame- or event-level ground truth exists.",
        "windows": {"scored": int(len(y)), "dropped_low_coverage": scored["dropped"], "total": scored["total"],
                    "participants": int(len(set(participants))), "videos": int(len(set(videos)))},
        "window_level": summarize(y, pred),
        "participant_variability": participant_breakdown(y, pred, participants),
        "confidence_intervals_95_participant_bootstrap": {
            "macro_f1": cluster_bootstrap(y, pred, participants, macro_f1, cfg["evaluation"]["bootstrap_samples"], cfg["seed"]),
            "balanced_accuracy": cluster_bootstrap(y, pred, participants, balanced_accuracy, cfg["evaluation"]["bootstrap_samples"], cfg["seed"])},
        "calibration_on_this_split": {"temperature": bundle.temperature, "ece_uncalibrated": ece(raw_probs, y), "ece_after_temperature": ece(probs, y),
                                      "nll_uncalibrated": nll(raw_probs, y), "nll_after_temperature": nll(probs, y),
                                      "note": "Agreement with weak video labels, not with ground-truth drowsiness events."},
    }
    video_ids = sorted(set(videos))
    vy, vp = [], []
    for v in video_ids:
        sel = videos == v
        vy.append(int(y[sel][0]))
        vp.append(int(probs[sel].mean(axis=0).argmax()))
    result["video_level_mean_probability"] = summarize(np.array(vy), np.array(vp))
    result["video_level_mean_probability"]["note"] = f"{len(video_ids)} videos from {len(set(participants))} participants: very small n; wide uncertainty."
    result["latency"] = measure_latency(bundle, scored, rows, cfg["evaluation"]["latency_windows"], cfg["seed"])
    result["environment"] = environment()
    result["previous_evaluations_of_this_split"] = repeats
    dump_json(run_dir / f"metrics_{split}.json", result)
    (run_dir / f"predictions_{split}.csv").write_text(
        "participant_id,video_id,t0,t1,label,pred,p_alert,p_low_vigilance,p_drowsy\n" +
        "\n".join(f"{m['participant_id']},{m['video_id']},{m['t0']:.2f},{m['t1']:.2f},{CLASS_NAMES[m['y']]},{CLASS_NAMES[p]},{q[0]:.4f},{q[1]:.4f},{q[2]:.4f}"
                  for m, p, q in zip(scored["meta"], pred, probs)), encoding="utf-8")
    (run_dir / f"report_{split}.md").write_text(render_report(result), encoding="utf-8")
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps({"split": split, "time": result["evaluated"], "weights": bundle.meta["weights_sha256"][:12]}) + "\n")
    if record_in_bundle and split == "test" and bundle.meta["calibration"].get("method") == "temperature":
        meta = json.loads((run_dir / "bundle.json").read_text(encoding="utf-8"))
        meta["validation_status"].update(heldout_evaluated=True, heldout_calibration={
            "ece": result["calibration_on_this_split"]["ece_after_temperature"], "n_participants": int(len(set(participants))),
            "n_windows": int(len(y)), "evaluated": result["evaluated"], "labels": "weak video-level"})
        dump_json(run_dir / "bundle.json", meta)
    return result


def render_report(r):
    w, cm = r["window_level"], r["window_level"]["confusion_matrix"]
    ci = r["confidence_intervals_95_participant_bootstrap"]
    fmt = lambda c: "n/a" if c.get("low") is None else f"{c['low']:.3f}-{c['high']:.3f}"
    lines = [f"# Evaluation report: {r['split']} split", "",
             f"Model: {r['model_type']} | evaluated {r['evaluated']} | previous looks at this split: {r['previous_evaluations_of_this_split']}", "",
             f"**Label semantics:** {r['label_semantics']}", "",
             f"Windows scored {r['windows']['scored']} of {r['windows']['total']} (dropped for low tracking coverage: {r['windows']['dropped_low_coverage']}); "
             f"{r['windows']['participants']} participants, {r['windows']['videos']} videos.", "",
             "## Window-level (agreement with weak video labels)", "",
             f"- macro-F1 **{w['macro_f1']:.3f}** (95% participant-bootstrap CI {fmt(ci['macro_f1'])})",
             f"- balanced accuracy **{w['balanced_accuracy']:.3f}** (CI {fmt(ci['balanced_accuracy'])}); plain accuracy {w['accuracy']:.3f}", "",
             "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |" for k, v in w["per_class"].items()]
    lines += ["", "Confusion matrix (rows = true, cols = predicted; order alert, low_vigilance, drowsy):", "", "```"]
    lines += [" ".join(f"{x:6d}" for x in row) for row in cm["matrix"]] + ["```", ""]
    pv = r["participant_variability"]
    lines += ["## Participant-level variability", "",
              f"Per-participant window accuracy: mean {pv['accuracy_spread']['mean']:.3f}, std {pv['accuracy_spread']['std'] if pv['accuracy_spread']['std'] is None else round(pv['accuracy_spread']['std'], 3)}, "
              f"min {pv['accuracy_spread']['min']:.3f}, max {pv['accuracy_spread']['max']:.3f} over {pv['accuracy_spread']['n']} participants.", "",
              "## Video level", "",
              f"macro-F1 {r['video_level_mean_probability']['macro_f1']:.3f} on {r['video_level_mean_probability']['n']} videos. {r['video_level_mean_probability']['note']}", "",
              "## Calibration on this split", "",
              f"ECE {r['calibration_on_this_split']['ece_uncalibrated']:.3f} (raw softmax) -> {r['calibration_on_this_split']['ece_after_temperature']:.3f} (temperature "
              f"{r['calibration_on_this_split']['temperature']:.2f}). {r['calibration_on_this_split']['note']}", "", "## Latency", ""]
    for name, key in (("features+model per window", "window_scoring_features_plus_model"), ("predictor.update per frame", "per_frame_predictor_update")):
        s = r["latency"].get(key)
        if s:
            lines.append(f"- {name}: p50 {s['p50_ms']:.2f} ms, p95 {s['p95_ms']:.2f} ms, max {s['max_ms']:.2f} ms (n={s['n']}) on {r['latency']['machine']['processor'] or r['latency']['machine']['platform']}")
    lines += ["", f"_{r['latency']['excludes']}._", "",
              "## Limits", "",
              "- Each participant recorded one video per class in separate sessions; session/camera/lighting differences can be learned instead of drowsiness. Participant-disjoint splits do not remove this.",
              "- No event annotations exist: nothing here supports claims about blink, yawn or microsleep detection accuracy.",
              "- Not validated for automotive or safety use."]
    return "\n".join(lines) + "\n"
