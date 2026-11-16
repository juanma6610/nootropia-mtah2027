"""Players only dead ball opinion: P(dead) from the players the camera sees, with no ball input.

The dead-ball BiGRU (dead_model.py) reads the ball (seen, gap length, out of the pitch), so where the tracker loses the
ball its P(dead) leans on the gap, uses it only where the ball is unseen: there P(dead) is the mean of the two, elsewhere the
dead-ball BiGRU's alone (gated()).

Model (SetGRU): Deep Sets over the players, then the dead-ball BiGRU's temporal part. Per player (x, y centred on the
pitch, vx, vy, speed, team +-1) -> phi (6 -> 64 -> 64, ReLU) -> masked mean and max over players, plus an empty flag ->
BiGRU 64 x 2 at 5 fps; team-swap averaged. A slot counts where it is tracked, its 0.4 s velocity is under 12 m/s (faster:
a slot jump) and it projects within MARGIN_PX of the frame (BAR-ATH: its own homography; a benchmark copy:
training/freeze.py's virtual camera), so BAR-ATH's frozen off-screen players stay out. Trained on the 7 DFL games and their noise
copies (training/train_players_dead.py, deterministic on CPU); models/players_dead.pt.

"""

import os

import numpy as np

from .loaders import CACHE_DIR, TEAM_DIR, _cached  # first: puts the upstream modules on sys.path
import baseline  # noqa: E402  (upstream starter/baseline.py)

# Params
MODEL_PATH = os.path.join(TEAM_DIR, "models", "players_dead.pt")
STEP = 5                 # frames: 5 fps, as the dead-ball BiGRU
K = 24                   # players per step at most (the on-screen ones first)
VEL_LAG = 5              # frames each side for the velocity: 0.4 s
MAX_SPEED = 12.0         # m/s: faster is a slot jumping to another player
FPS = 25
HIDDEN, LAYERS, PHI = 64, 2, 64
THREADS = 8              # torch CPU threads; fixed so that predictions are reproducible bit for bit
TRAIN_THREADS = 4        # torch CPU threads for training, as the shipped model was fitted


W_PX, H_PX = 1920, 1080   # the tactical video frame
MARGIN_PX = 250           # px beyond the frame a player still counts as seen (training/freeze.py's MARGIN_PX)


def project(H, xn, yn):
    """Pixels (u, v) of normalised pitch points; H [N, 3, 3], xn / yn [N, S] (as training/freeze.py's project())."""
    d = H[:, 2, 0, None] * xn + H[:, 2, 1, None] * yn + H[:, 2, 2, None]
    u = (H[:, 0, 0, None] * xn + H[:, 0, 1, None] * yn + H[:, 0, 2, None]) / d
    v = (H[:, 1, 0, None] * xn + H[:, 1, 1, None] * yn + H[:, 1, 2, None]) / d
    return u, v


def onscreen(H, x, y, pitch):
    """bool[n, S]: tracked and within MARGIN_PX of the frame."""
    length, width = pitch
    u, v = project(H, x / length, y / width)
    edge = np.minimum(np.minimum(u, W_PX - u), np.minimum(v, H_PX - v))
    with np.errstate(invalid="ignore"):
        return np.isfinite(x) & (edge >= -MARGIN_PX)


def set_arrays(x, y, teams, on_idx, pitch):
    """(P float32[steps, K, 6], M bool[steps, K]) at STEP resolution; on_idx: the on-screen mask at those frames."""
    length, width = pitch
    n = len(x)
    idx = np.arange(0, n, STEP)
    lo, hi = np.clip(idx - VEL_LAG, 0, n - 1), np.clip(idx + VEL_LAG, 0, n - 1)
    dt = np.maximum(hi - lo, 1)[:, None] / FPS
    with np.errstate(invalid="ignore"):
        vx, vy = (x[hi] - x[lo]) / dt, (y[hi] - y[lo]) / dt
        sp = np.hypot(vx, vy)
        ok = on_idx & np.isfinite(sp) & (sp < MAX_SPEED)
    team = np.where(teams[idx] == baseline.HOME, 1.0, np.where(teams[idx] == baseline.AWAY, -1.0, 0.0))
    feats = np.stack([x[idx] / length - 0.5, y[idx] / width - 0.5, vx / 5, vy / 5, sp / 5, team], -1)
    order = np.argsort(~ok, axis=1, kind="stable")[:, :K]
    P = np.take_along_axis(feats, order[:, :, None], axis=1)
    M = np.take_along_axis(ok, order, axis=1)
    if P.shape[1] < K:
        pad = K - P.shape[1]
        P = np.concatenate([P, np.zeros((len(P), pad, 6))], 1)
        M = np.concatenate([M, np.zeros((len(M), pad), bool)], 1)
    return np.where(M[:, :, None], np.nan_to_num(P), 0.0).astype(np.float32), M


def _net():
    import torch
    from torch import nn

    class SetGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(6, PHI), nn.ReLU(), nn.Linear(PHI, PHI), nn.ReLU())
            self.inp = nn.Sequential(nn.Linear(2 * PHI + 1, HIDDEN), nn.ReLU())
            self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=LAYERS, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * HIDDEN, 1)

        def forward(self, P, M):            # P [B, T, K, 6], M [B, T, K]
            e = self.phi(P)                  # >= 0 (ReLU): a masked max is the max of e * mask
            w = M.unsqueeze(-1)
            n = w.sum(2)
            pooled = torch.cat([(e * w).sum(2) / n.clamp(min=1), (e * w).amax(2), (n == 0).float()], -1)
            h, _ = self.gru(self.inp(pooled))
            return self.out(h).squeeze(-1)
    return SetGRU()


def setup(seed=0, threads=THREADS):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def load(path=MODEL_PATH):
    import torch
    net = _net()
    net.load_state_dict(torch.load(path))
    return net


def predict(net, q, runs):
    """float[n frames]: P(dead) per frame, each run on its own (swap-averaged), the rest from the whole sequence."""
    import torch
    setup()
    net.eval()
    n = q["n"]
    idx = np.arange(0, n, STEP)
    out = np.full(n, np.nan)
    with torch.no_grad():
        for first, stop in list(runs) + [(0, n)]:
            a, b = -(-first // STEP), -(-stop // STEP)
            if b - a < 2:
                continue
            P, M = q["P"][a:b][None].copy(), torch.from_numpy(q["M"][a:b][None].astype(np.float32))
            p = torch.sigmoid(net(torch.from_numpy(P), M))[0].numpy()
            P[..., 5] *= -1
            p = (p + torch.sigmoid(net(torch.from_numpy(P), M))[0].numpy()) / 2
            frames = np.arange(first, stop)
            vals = np.interp(frames, idx[a:b], p)
            out[first:stop] = np.where(np.isnan(out[first:stop]), vals, out[first:stop])
    return out


def bar_ath_inputs():
    """BAR-ATH's player sets ({"P", "M", "n"}) and ball-seen mask (bool[n]). Cached."""
    def build():
        import load_homography  # noqa: E402  (upstream starter/load_homography.py)
        from . import ballnet
        x, y, teams, ball, pitch = ballnet.bar_ath_arrays()
        x, y = x.astype(float), y.astype(float)
        idx = np.arange(0, len(x), STEP)
        H, _, valid = load_homography.load()
        on = onscreen(H[idx], x[idx], y[idx], pitch)
        on[~valid[idx]] = np.isfinite(x[idx][~valid[idx]])  # no homography: every tracked slot
        P, M = set_arrays(x, y, teams, on, pitch)
        return {"P": P, "M": M, "n": len(x)}, np.isfinite(ball[:, 0])
    return _cached(os.path.join(CACHE_DIR, "players_dead_inputs.pkl"), build)


def gated(p_ship, p_players, seen):
    """P(dead): the dead-ball BiGRU's where the ball is seen, the mean with the players-only one where it is not."""
    p_players = np.where(np.isfinite(p_players), p_players, p_ship)
    return np.where(seen, p_ship, (p_ship + p_players) / 2)


def bar_ath(p_ship):
    """BAR-ATH's P(dead) from the dead-ball BiGRU's (p_ship), the players-only opinion added where the ball is unseen."""
    from .decode import bar_ath_runs  # here: decode is line 3
    q, seen = bar_ath_inputs()
    return gated(p_ship, predict(load(), q, bar_ath_runs()), seen)
