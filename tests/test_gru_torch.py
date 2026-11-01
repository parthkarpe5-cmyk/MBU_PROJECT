"""GRU training: needs PyTorch. SKIPPED in the build sandbox (no torch); run it on Colab/your GPU box."""
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from dms.learned import Bundle, gru_logits
from research.config import load_config
from research.synthetic import write_synthetic_cache
from research.temporal import make_net, numpy_weights, train_gru

CFG = load_config(None, {"features": {"window_seconds": 30, "stride_seconds": 10},
                         "gru": {"window_seconds": 30.0, "hidden_grid": [8], "max_epochs": 2, "patience": 2, "batch_size": 32, "device": "cpu"}})


def test_numpy_forward_matches_torch():
    net = make_net(torch, 11, 8, 0.0).eval()
    x = torch.randn(4, 30, 11)
    mask = (torch.rand(4, 30) > .2).float()
    ref = net(x, mask).detach().numpy()
    w = numpy_weights(net)
    ours = np.array([gru_logits(w, x[i].numpy().astype(np.float64), mask[i].numpy()) for i in range(4)])
    assert np.abs(ref - ours).max() < 1e-4


def test_train_resume_and_bundle(tmp_path):
    rows = write_synthetic_cache(tmp_path / "cache", folds=5, per_fold=3, seconds=150)
    logs = []
    train_gru(CFG, rows, tmp_path / "cache", tmp_path / "run", "sha", synthetic=True, max_epochs=1, log=logs.append)
    train_gru(CFG, rows, tmp_path / "cache", tmp_path / "run", "sha", synthetic=True, max_epochs=2, log=logs.append)
    assert any("resumed at epoch 1" in l for l in logs)
    bundle = Bundle(tmp_path / "run")
    assert bundle.model_type == "gru" and json.loads((tmp_path / "run" / "bundle.json").read_text())["training"]["test_split_used"] is False
