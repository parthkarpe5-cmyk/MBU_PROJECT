"""Small GRU over 1-second feature bins (PyTorch; training only, not needed at runtime).

NOT EXECUTED IN THE BUILD SANDBOX (no PyTorch there). It is checked by py_compile and by
tests that need torch (skipped here). Its export is gated by a mandatory torch-vs-numpy parity
check against ``dms.learned.gru_logits``, the exact code the app runs.

Hygiene: preprocessing fit on train only; early stopping + hyperparameter choice on validation only;
the test split is never loaded; per-epoch shuffling seeded by (seed, epoch) and a checkpoint after
every epoch so Colab disconnects resume exactly.
"""
import os
from pathlib import Path

import numpy as np

from dms.features import SequencePreprocessor
from dms.learned import gru_logits
from .bundle import write_bundle
from .config import dump_json, environment, feature_config, set_seed, sha256_json
from .metrics import fit_temperature, macro_f1, summarize
from .windows import build_windows, subset, summarize_windows


def require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for GRU training: pip install -r requirements-research.txt "
                         "(CPU or CUDA build). It is not needed to run the app.") from exc


def make_net(torch, in_dim, hidden, dropout):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = hidden
            self.cell = nn.GRUCell(in_dim, hidden)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden, 3)

        def forward(self, x, mask):
            h = x.new_zeros(x.shape[0], self.hidden)
            for step in range(x.shape[1]):
                h_new = self.cell(x[:, step], h)
                m = mask[:, step:step + 1]
                h = m * h_new + (1.0 - m) * h      # unobserved bins carry state; never treated as closed eyes/zeros
            return self.out(self.drop(h))
    return Net()


def numpy_weights(model):
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    return {"w_ih": sd["cell.weight_ih"], "w_hh": sd["cell.weight_hh"], "b_ih": sd["cell.bias_ih"],
            "b_hh": sd["cell.bias_hh"], "w_out": sd["out.weight"], "b_out": sd["out.bias"]}


def _atomic_save(torch, obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _predict(torch, model, X, M, device, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(X[i:i + batch].to(device), M[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def fit_one(torch, hidden, cfg, data, sub_dir, signature, resume, log, max_epochs=None):
    """Train one hyperparameter setting; resumable. Returns dict(best_f1, best_epoch, best_path)."""
    g = cfg["gru"]
    device = torch.device("cuda" if g["device"] == "auto" and torch.cuda.is_available() else "cpu" if g["device"] == "auto" else g["device"])
    Xtr, Mtr, ytr, Xva, Mva, yva = data
    set_seed(cfg["seed"])
    model = make_net(torch, Xtr.shape[2], hidden, g["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=g["lr"], weight_decay=g["weight_decay"])
    counts = np.bincount(ytr.numpy(), minlength=3).astype(np.float64)
    weights = torch.tensor(len(ytr) / (3 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    state = dict(epoch=0, best_f1=-1.0, best_epoch=-1, bad=0, history=[], done=False)
    ckpt, best_path = Path(sub_dir) / "last.pt", Path(sub_dir) / "best.pt"
    if resume and ckpt.is_file():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        if saved["signature"] == signature:
            model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            state = saved["state"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            log(f"[h={hidden}] resumed at epoch {state['epoch']} (best val macro-F1 {state['best_f1']:.4f})")
        else:
            log(f"[h={hidden}] checkpoint signature differs (config/data changed); starting fresh")
    limit = max_epochs or g["max_epochs"]
    n = len(ytr)
    while state["epoch"] < limit and not state["done"]:
        model.train()
        shuffle = torch.Generator()
        shuffle.manual_seed(cfg["seed"] * 100003 + state["epoch"])
        order = torch.randperm(n, generator=shuffle)
        total = 0.0
        for i in range(0, n, g["batch_size"]):
            idx = order[i:i + g["batch_size"]]
            logits = model(Xtr[idx].to(device), Mtr[idx].to(device))
            loss = torch.nn.functional.cross_entropy(logits, ytr[idx].to(device), weight=weights)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        val_f1 = macro_f1(yva.numpy(), _predict(torch, model, Xva, Mva, device).argmax(1))
        state["history"].append(dict(epoch=state["epoch"], train_loss=total / n, val_macro_f1=val_f1))
        if val_f1 > state["best_f1"]:
            state.update(best_f1=val_f1, best_epoch=state["epoch"], bad=0)
            _atomic_save(torch, {"model": model.state_dict(), "epoch": state["epoch"]}, best_path)
        else:
            state["bad"] += 1
        state["epoch"] += 1
        state["done"] = state["bad"] >= g["patience"]
        _atomic_save(torch, {"model": model.state_dict(), "optimizer": opt.state_dict(), "state": state,
                             "torch_rng": torch.get_rng_state(), "signature": signature}, ckpt)
        log(f"[h={hidden}] epoch {state['epoch']}: loss {total / n:.4f} val macro-F1 {val_f1:.4f} (best {state['best_f1']:.4f} @ {state['best_epoch']})")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    return dict(hidden=hidden, best_f1=state["best_f1"], best_epoch=state["best_epoch"], model=model, device=device,
                history=state["history"])


def train_gru(cfg, rows, cache_dir, run_dir, manifest_sha=None, resume=True, synthetic=False, max_epochs=None, log=print):
    torch = require_torch()
    set_seed(cfg["seed"])
    run_dir = Path(run_dir)
    g = cfg["gru"]
    fcfg = feature_config(cfg, window_seconds=float(g["window_seconds"]))
    wins = build_windows(rows, cache_dir, fcfg, "sequence", splits=("train", "val"), log=log)
    train, val = subset(wins, "train"), subset(wins, "val")
    if len(set(train["y"])) < 3 or len(val["y"]) == 0:
        raise RuntimeError("Need all 3 classes in train and a non-empty validation split")
    pre = SequencePreprocessor().fit(train["V"], train["M"])
    tensor = lambda a, dtype=torch.float32: torch.tensor(np.asarray(a), dtype=dtype)
    data = (tensor(pre.transform(train["V"], train["M"])), tensor(train["M"]), tensor(train["y"], torch.long),
            tensor(pre.transform(val["V"], val["M"])), tensor(val["M"]), tensor(val["y"], torch.long))
    log(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}; train windows {len(train['y'])}, val windows {len(val['y'])}")
    results = []
    for hidden in g["hidden_grid"]:
        signature = sha256_json({"hidden": hidden, "gru": g, "fcfg": fcfg.to_dict(), "seed": cfg["seed"], "n_train": len(train["y"]), "manifest": manifest_sha})
        results.append(fit_one(torch, hidden, cfg, data, run_dir / "gru" / f"h{hidden}", signature, resume, log, max_epochs))
    best = max(results, key=lambda r: r["best_f1"])           # first max wins ties
    model, device = best["model"], best["device"]
    val_logits = _predict(torch, model, data[3], data[4], device)
    weights = numpy_weights(model)
    # Mandatory parity gate: the numpy forward the app uses must reproduce the torch model.
    probe = min(64, len(val["y"]))
    reference = _predict(torch, model, data[3][:probe], data[4][:probe], device)
    ours = np.array([gru_logits(weights, data[3][i].numpy().astype(np.float64), val["M"][i]) for i in range(probe)])
    diff = float(np.abs(reference - ours).max())
    if not diff < 1e-3:
        raise RuntimeError(f"torch/numpy GRU parity FAILED (max abs diff {diff:.2e}); refusing to write a bundle")
    log(f"torch/numpy parity max abs logit diff: {diff:.2e}")
    T = fit_temperature(val_logits, val["y"])
    training = {
        "dataset": "SYNTHETIC SMOKE FIXTURE (not UTA-RLDD)" if synthetic else "UTA-RLDD", "synthetic": bool(synthetic), "seed": cfg["seed"],
        "selected": {"hidden": best["hidden"], "best_epoch": best["best_epoch"], "window_seconds": g["window_seconds"]},
        "selection_split": "validation", "preprocessing_fit_split": "train", "calibration_fit_split": "validation",
        "train": summarize_windows(train), "validation": summarize_windows(val), "manifest_sha256": manifest_sha,
        "config_sha256": sha256_json(cfg), "test_split_used": False, "torch_numpy_parity_max_abs_diff": diff,
        "device": str(device), "window_build_stats": {k: v for k, v in wins["stats"].items() if k != "videos_missing_cache"},
    }
    meta = write_bundle(run_dir, "gru", weights, pre.to_dict(), fcfg, val_logits, val["y"], T, training)
    dump_json(run_dir / "selection.json", {"results": [{k: r[k] for k in ("hidden", "best_f1", "best_epoch", "history")} for r in results],
                                           "rule": "max validation macro-F1"})
    dump_json(run_dir / "config.json", cfg)
    dump_json(run_dir / "env.json", environment())
    vm = summarize(val["y"], val_logits.argmax(1))
    vm["note"] = "VALIDATION metrics (used for early stopping, selection, calibration): optimistic."
    dump_json(run_dir / "metrics_val.json", vm)
    return meta
