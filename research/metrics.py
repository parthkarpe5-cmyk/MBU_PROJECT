"""Metrics with no hidden dependencies (numpy only)."""
import numpy as np

CLASS_NAMES = ("alert", "low_vigilance", "drowsy")


def confusion_matrix(y, pred, n=3):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (np.asarray(y, int), np.asarray(pred, int)), 1)
    return cm


def per_class(cm):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)      # zero_division -> 0
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1, support


def macro_f1(y, pred, n=3):
    return float(per_class(confusion_matrix(y, pred, n))[2].mean())


def balanced_accuracy(y, pred, n=3):
    cm = confusion_matrix(y, pred, n)
    support = cm.sum(axis=1)
    present = support > 0
    return float((np.diag(cm)[present] / support[present]).mean()) if present.any() else float("nan")


def summarize(y, pred, names=CLASS_NAMES):
    cm = confusion_matrix(y, pred, len(names))
    p, r, f, s = per_class(cm)
    return {
        "n": int(len(y)), "accuracy": float(np.mean(np.asarray(y) == np.asarray(pred))) if len(y) else float("nan"),
        "macro_f1": float(f.mean()), "balanced_accuracy": balanced_accuracy(y, pred, len(names)),
        "per_class": {n: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i])) for i, n in enumerate(names)},
        "confusion_matrix": {"rows_true_cols_pred": list(names), "matrix": cm.tolist()},
    }


def nll(probs, y):
    p = np.clip(np.asarray(probs)[np.arange(len(y)), np.asarray(y, int)], 1e-12, 1.0)
    return float(-np.log(p).mean())


def ece(probs, y, bins=15):
    """Top-label expected calibration error against the (weak) window labels."""
    probs, y = np.asarray(probs), np.asarray(y, int)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return float(total)


def fit_temperature(logits, y):
    """Single-parameter temperature scaling on VALIDATION logits. Deterministic grid + refinement."""
    logits, y = np.asarray(logits, float), np.asarray(y, int)

    def loss(T):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    best = grid[int(np.argmin([loss(T) for T in grid]))]
    fine = np.linspace(best * 0.9, best * 1.1, 200)
    return float(fine[int(np.argmin([loss(T) for T in fine]))])


def auroc(scores, positive):
    """Mann-Whitney AUROC with average ranks for ties. positive: boolean array."""
    scores, positive = np.asarray(scores, float), np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def participant_breakdown(y, pred, participants, names=CLASS_NAMES):
    """Window-level results per participant plus spread across participants (not event-level)."""
    y, pred, participants = np.asarray(y), np.asarray(pred), np.asarray(participants)
    rows = {}
    for p in sorted(set(participants)):
        sel = participants == p
        recalls = {}
        for i, n in enumerate(names):
            c = sel & (y == i)
            recalls[n] = float((pred[c] == i).mean()) if c.any() else None
        present = [v for v in recalls.values() if v is not None]
        rows[p] = {"n_windows": int(sel.sum()), "accuracy": float((pred[sel] == y[sel]).mean()),
                   "balanced_accuracy": float(np.mean(present)) if present else None, "recall_by_class": recalls}
    accs = np.array([v["accuracy"] for v in rows.values()])
    bals = np.array([v["balanced_accuracy"] for v in rows.values() if v["balanced_accuracy"] is not None])
    spread = lambda a: dict(mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a) > 1 else None,
                            min=float(a.min()), max=float(a.max()), n=int(len(a))) if len(a) else None
    return {"per_participant": rows, "accuracy_spread": spread(accs), "balanced_accuracy_spread": spread(bals)}


def cluster_bootstrap(y, pred, participants, fn, samples=1000, seed=0, alpha=0.05):
    """Percentile CI from resampling PARTICIPANTS (windows within a participant are correlated)."""
    y, pred, participants = np.asarray(y), np.asarray(pred), np.asarray(participants)
    ids = sorted(set(participants))
    if len(ids) < 2:
        return {"low": None, "high": None, "note": "need >= 2 participants"}
    index = {p: np.flatnonzero(participants == p) for p in ids}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        sel = np.concatenate([index[ids[i]] for i in pick])
        values.append(fn(y[sel], pred[sel]))
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"low": float(lo), "high": float(hi), "samples": samples, "unit": "participants", "level": 1 - alpha}


def latency_stats(ms):
    ms = np.asarray(ms, float)
    if not len(ms):
        return None
    return {"n": int(len(ms)), "mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)), "max_ms": float(ms.max())}
