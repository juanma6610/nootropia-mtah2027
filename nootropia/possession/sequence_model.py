"""Sequence model: a small bidirectional GRU, trained on DFL, for one of two targets.

    "home"  P(home owns the ball). Reads the ownership_model.py inputs; the loss covers live
            frames only; home/away swap augmentation, prediction averaged with the swapped sequence.
    "dead"  P(dead ball) (dead.sequence_inputs()). The loss covers every graded frame;
            no swap (the target does not depend on the teams).

About +/- 5 s of context. Labels only from other games.

Runs at `step`-frame resolution (25 fps / step), trains on `window`-step windows and predicts whole
halves at once; probabilities are interpolated back to every frame. CPU only, seeded.

Every setting lives in a config dict (DEFAULT, overridden per call) saved with the model.
training/train_ownership.py and training/train_dead.py fit the two adopted models.
"""

import os

import numpy as np
import torch
from torch import nn

from .loaders import TEAM_DIR  # first: puts the upstream modules on sys.path
from . import ownership

# Params
MODEL_PATH = os.path.join(TEAM_DIR, "models", "sequence.pt")
DEFAULT = {
    "step": 5,             # frames per model step: 5 fps
    "window": 50,          # steps per training window: 10 s
    "stride": 25,          # steps between training windows
    "hidden": 64,
    "layers": 2,
    "epochs": 6,
    "batch": 64,
    "learning_rate": 1e-3,
    "seed": 0,
    "target": "home",      # "home" or "dead" (module docstring)
}
THREADS = 8


def _config(config):
    out = dict(DEFAULT)
    out.update(config or {})
    return out


class Net(nn.Module):
    def __init__(self, n_inputs, hidden=DEFAULT["hidden"], layers=DEFAULT["layers"]):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h).squeeze(-1)


def _setup(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


class Scaler:
    """Median / IQR per column from the training rows (fitted in training/bigru.py); NaN -> 0 plus a missing
    flag per column."""

    def transform(self, table):
        values = (table[self.columns].to_numpy(dtype=float) - self.median) / self.scale
        missing = np.isnan(values)
        return np.concatenate([np.nan_to_num(np.clip(values, -10, 10)), missing], axis=1).astype(np.float32)


def _steps(table, step):
    """Every step-th frame of a table."""
    return table.iloc[::step].reset_index(drop=True)


def _predict_steps(net, scaler, table, step):
    """float[steps]: P(target) per step."""
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(scaler.transform(_steps(table, step)))[None])).numpy()[0]


def _per_frame(n, step, values):
    frames = np.arange(n)
    return np.interp(frames, frames[::step][:len(values)], values)


def predict(net, scaler, table, runs=None, config=None):
    """float[N]: P(target) per frame ("home": averaged with the swapped sequence), each run (half)
    predicted on its own; frames outside runs get the whole-table prediction."""
    cfg = _config(config)
    _setup(cfg["seed"])
    net.eval()
    n = len(table)
    out = np.full(n, np.nan)
    pieces = [(first, stop) for first, stop in (runs or [(0, n)])]
    if runs:
        pieces.append((0, n))  # fills the frames outside the runs
    for first, stop in pieces:
        part = table.iloc[first:stop].reset_index(drop=True)
        p = _predict_steps(net, scaler, part, cfg["step"])
        if cfg["target"] == "home":
            p = (p + 1 - _predict_steps(net, scaler, ownership.swap(part), cfg["step"])) / 2
        p = _per_frame(len(part), cfg["step"], p)
        target = slice(first, stop)
        out[target] = np.where(np.isnan(out[target]), p, out[target])
    return out


def load(path=MODEL_PATH):
    """(Net, Scaler, config). Settings a model was saved without get DEFAULT's."""
    saved = torch.load(path, weights_only=False)
    cfg = _config(saved.get("config"))
    net = Net(saved["n_inputs"], cfg["hidden"], cfg["layers"])
    net.load_state_dict(saved["state"])
    scaler = Scaler()
    scaler.columns, scaler.median, scaler.scale = saved["columns"], saved["median"], saved["scale"]
    return net, scaler, cfg
