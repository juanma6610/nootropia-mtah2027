"""Training of possession/sequence_model.py's BiGRU: windows, loss, fit and save, seeded and deterministic on CPU."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from possession import ownership
from possession.sequence_model import Net, Scaler, _config, _setup, _steps


def fit_scaler(table):
    """A sequence_model.Scaler fitted on training rows: median / IQR per column."""
    scaler = Scaler()
    scaler.columns = list(table.columns)
    values = table.to_numpy(dtype=float)
    scaler.median = np.nanmedian(values, axis=0)
    iqr = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    scaler.scale = np.where(iqr > 1e-6, iqr, 1.0)
    return scaler


def _targets(lab, sign, cfg):
    """(y float32[steps], mask bool[steps]: the steps the loss counts) for one (possibly team-swapped)
    sequence."""
    if cfg["target"] == "dead":
        return (lab == "D").astype(np.float32), lab != "X"
    return (lab == ("H" if sign == 1 else "A")).astype(np.float32), np.isin(lab, ["H", "A"])


def training_windows(tables_labels, scaler, cfg):
    """(x float[W, window, F], y float[W, window], mask bool[W, window]) from (table, labels) pairs;
    for "home" each pair also team-swapped and the loss counts live steps only."""
    step, window, stride = cfg["step"], cfg["window"], cfg["stride"]
    xs, ys, ms = [], [], []
    for table, labels in tables_labels:
        lab = labels[::step]
        pairs = ((table, 1),) if cfg["target"] == "dead" else ((table, 1), (ownership.swap(table), -1))
        for t, sign in pairs:
            x = scaler.transform(_steps(t, step))
            y, m = _targets(lab, sign, cfg)
            for s in range(0, len(x) - window + 1, stride):
                if m[s:s + window].any():
                    xs.append(x[s:s + window]), ys.append(y[s:s + window]), ms.append(m[s:s + window])
    return np.stack(xs), np.stack(ys), np.stack(ms)


def fit(tables_labels, config=None):
    """(Net, Scaler, config) trained on (input table, labels) pairs."""
    cfg = _config(config)
    _setup(cfg["seed"])
    scaler = fit_scaler(pd.concat([_steps(t, cfg["step"]) for t, _ in tables_labels], ignore_index=True))
    x, y, m = training_windows(tables_labels, scaler, cfg)
    net = Net(x.shape[2], cfg["hidden"], cfg["layers"])
    optimiser = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    x, y, m = (torch.from_numpy(v) for v in (x, y, m))
    generator = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg["batch"]):
            idx = order[start:start + cfg["batch"]]
            weight = m[idx].sum().clamp(min=1)
            loss = (loss_fn(net(x[idx]), y[idx]) * m[idx]).sum() / weight
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return net, scaler, cfg


def save(net, scaler, config, path):
    torch.save({"state": net.state_dict(), "n_inputs": net.inp[0].in_features, "columns": scaler.columns,
                "median": scaler.median, "scale": scaler.scale, "config": _config(config)}, path)
