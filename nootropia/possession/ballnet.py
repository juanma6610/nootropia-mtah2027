"""Ballnet: where the ball is and who has it, inferred from the players alone (after Kim et al. 2023,
"Ball trajectory inference from multi-agent sports contexts using set transformers and hierarchical
Bi-LSTMs").

Per 5 fps step the tracked players are a set of tokens (x, y, vx, vy, speed, team +1 / -1); there is no
ball input. Anonymous track slots are fine: the set is permutation invariant.

    token MLP -> one self-attention layer over the set (padding masked)
    -> pooled overall and per team -> BiGRU 1 -> team possession logit          (Kim: team level)
    -> [step, BiGRU 1] -> BiGRU 2 -> pointer over the players (who is on the ball) (Kim: player level)
                                  -> ball x, y                                   (Kim: ball level)

Trained on the 7 DFL benchmark games x 3 noise copies (training/train_ballnet.py), never on BAR-ATH:
targets from the clean DFL source (the ball, read from the raw positions XML; team possession from DFL's
BallPossession flag; the possessor = the player nearest the clean ball within 1.5 m). Each training
window is, at random, the copy as built or the copy with off-screen players frozen (training/freeze.py); plus team swap and pitch flips.

Its outputs enter the ownership BiGRU as 8 extra inputs (COLUMNS, H_ / A_ pairs so ownership_model.swap()
exchanges them).

    p = bar_ath_columns()             # DataFrame of COLUMNS for BAR-ATH, models/ballnet.pt
"""

import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from .loaders import CACHE_DIR, FPS, TEAM_DIR, _cached  # first: puts the upstream modules on sys.path
from . import slots
import baseline  # noqa: E402  (upstream starter/baseline.py)

# Params
MODEL_PATH = os.path.join(TEAM_DIR, "models", "ballnet.pt")
STEP = 5                 # frames per model step: 5 fps
MAX_PLAYER_SPEED = 12.0  # m/s: faster is a slot jumping to another player
TOKEN_F = 6
CFG = {"d": 48, "heads": 4, "hidden": 64, "window": 50, "stride": 50, "epochs": 20, "batch": 32,
       "lr": 1e-3, "seed": 0, "w_ball": 1.0, "w_team": 1.0, "w_ptr": 0.5}
THREADS = 6              # the shipped nets were trained with 6 threads
COLUMNS = ["H_bi_p", "A_bi_p", "H_bi_ptr", "A_bi_ptr", "H_bi_d1", "A_bi_d1", "bi_gap", "bi_conf"]


# ---------------------------------------------------------------- tokens


def tokens(x, y, teams, pitch, step=STEP):
    """(tok float32[T, K, F], valid bool[T, K]) at 1 / step of the frames: the tracked players compacted
    to the front of each row. x, y metres float[N, S]; teams int8[N, S] (baseline codes, -1 ignore)."""
    length, width = pitch
    vx = slots.velocity(x.astype(np.float32), FPS, dtype=np.float32)
    vy = slots.velocity(y.astype(np.float32), FPS, dtype=np.float32)
    sp = np.hypot(vx, vy)
    bad = ~(sp <= MAX_PLAYER_SPEED)
    vx[bad], vy[bad], sp[bad] = 0, 0, 0
    x, y, vx, vy, sp, teams = (a[::step] for a in (x, y, vx, vy, sp, teams))
    ok = np.isfinite(x) & np.isfinite(y) & (teams >= 0)
    sign = np.where(teams == baseline.HOME, 1.0, -1.0)
    feat = np.stack([x / (length / 2) - 1, y / (width / 2) - 1, vx / 8, vy / 8, sp / 8, sign], -1).astype(np.float32)
    k = int(ok.sum(1).max())
    order = np.argsort(~ok, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(x))[:, None]
    tok = np.nan_to_num(feat[rows, order])
    valid = ok[rows, order]
    tok[~valid] = 0
    return tok, valid


# ---------------------------------------------------------------- model
class BallNet(nn.Module):
    def __init__(self, d=CFG["d"], heads=CFG["heads"], hidden=CFG["hidden"]):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(TOKEN_F, d), nn.ReLU(), nn.Linear(d, d))
        self.att = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.frame = nn.Sequential(nn.Linear(3 * d + 2, d), nn.ReLU())
        self.gru1 = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.team = nn.Linear(2 * hidden, 1)
        self.gru2 = nn.GRU(d + 2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.ball = nn.Linear(2 * hidden, 2)
        self.query = nn.Linear(2 * hidden, d)

    def forward(self, tok, valid):
        """tok [B, T, K, F], valid [B, T, K] -> (team logit [B, T], ball [B, T, 2], ptr logits [B, T, K])."""
        B, T, K, F = tok.shape
        t = tok.reshape(B * T, K, F)
        v = valid.reshape(B * T, K)
        v_att = v.clone()
        v_att[~v.any(1), 0] = True  # an all-padded row would give NaN attention
        h = self.att(self.emb(t), src_key_padding_mask=~v_att)
        home = (t[:, :, 5] > 0) & v
        away = (t[:, :, 5] < 0) & v

        def pool(m):
            w = m.float().unsqueeze(-1)
            return (h * w).sum(1) / w.sum(1).clamp(min=1)
        frame = torch.cat([pool(v), pool(home), pool(away), home.float().sum(1, keepdim=True) / 11,
                           away.float().sum(1, keepdim=True) / 11], -1)
        f = self.frame(frame).reshape(B, T, -1)
        g1, _ = self.gru1(f)
        g2, _ = self.gru2(torch.cat([f, g1], -1))
        q = self.query(g2).reshape(B * T, 1, -1)
        ptr = (h * q).sum(-1).masked_fill(~v, -1e4).reshape(B, T, K)
        return self.team(g1).squeeze(-1), self.ball(g2), ptr


def load(path=MODEL_PATH):
    net = BallNet()
    net.load_state_dict(torch.load(path))
    return net


def predict(net, tok, valid, pitch, chunk=1500):
    """Per step: (p_home float[T] team-swap averaged, ball metres float[T, 2], ptr_home float[T] = pointer
    mass on home players, ptr_max float[T] = the pointer's top probability)."""
    length, width = pitch
    torch.set_num_threads(THREADS)
    net.eval()
    outs = []
    with torch.no_grad():
        for a in range(0, len(tok), chunk):  # chunks with 100 steps of context each side, middle kept
            lo, hi = max(0, a - 100), min(len(tok), a + chunk + 100)
            t = torch.from_numpy(tok[lo:hi])[None]
            v = torch.from_numpy(valid[lo:hi])[None]
            tl, b, p = net(t, v)
            ts = t.clone()
            ts[..., 5] *= -1
            tl2, b2, _ = net(ts, v)
            ph = (torch.sigmoid(tl) + 1 - torch.sigmoid(tl2))[0] / 2
            bb = ((b + b2) / 2)[0]
            pr = torch.softmax(p, -1)[0]
            home = (t[0, :, :, 5] > 0) & v[0]
            sl = slice(a - lo, a - lo + min(chunk, len(tok) - a))
            outs.append((ph[sl].numpy(), bb[sl].numpy(), (pr * home).sum(-1)[sl].numpy(), pr.max(-1).values[sl].numpy()))
    ph, bb, pth, pmx = (np.concatenate(z) for z in zip(*outs))
    return ph, np.stack([(bb[:, 0] + 1) * length / 2, (bb[:, 1] + 1) * width / 2], -1), pth, pmx


def frame_features(net, x, y, teams, obs, pitch):
    """(DataFrame of COLUMNS, one row per frame; inferred ball float[N, 2] metres): ballnet's team head,
    pointer mass per team, each team's nearest player to the inferred ball, the inferred ball's distance
    to the tracked one (obs, NaN = not seen), the pointer's confidence."""
    tok, valid = tokens(x, y, teams, pitch)
    ph, ball, pth, pmx = predict(net, tok, valid, pitch)
    n = len(x)
    frames, steps = np.arange(n), np.arange(n)[::STEP][:len(ph)]

    def up(v):
        return np.interp(frames, steps, v)
    b = np.stack([up(ball[:, 0]), up(ball[:, 1])], -1)
    dist = np.hypot(x - b[:, :1], y - b[:, 1:])
    out = {"H_bi_p": up(ph), "A_bi_p": 1 - up(ph), "H_bi_ptr": up(pth), "A_bi_ptr": 1 - up(pth)}
    for name, code in (("H", baseline.HOME), ("A", baseline.AWAY)):
        d = np.where((teams == code) & np.isfinite(dist), dist, np.inf).min(1)
        out[name + "_bi_d1"] = np.where(np.isfinite(d), d, np.nan)
    out["bi_gap"] = np.hypot(*(obs - b).T)
    out["bi_conf"] = up(pmx)
    return pd.DataFrame(out).astype(np.float32), b


# ---------------------------------------------------------------- data
def model_hash(path=MODEL_PATH):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:12]


def bar_ath_arrays():
    """(x, y, teams, ball, pitch) of BAR-ATH: slot positions in metres, slot teams with the keeper fix."""
    from .loaders import load_frame_df
    from games import GAME, GAMES  # noqa: E402
    frame_df, home_ids, away_ids = load_frame_df()
    x, y = slots.slot_positions(frame_df, list(home_ids) + list(away_ids))
    teams = slots.slot_teams(frame_df, home_ids, away_ids, fix=slots.KEEPER_FIX)
    length, width = GAMES[GAME]["pitch"]
    ball = frame_df[["ball_x", "ball_y"]].to_numpy(dtype=float) * (length, width)
    return x, y, teams, ball, (length, width)


def bar_ath_columns():
    """DataFrame of COLUMNS for BAR-ATH from models/ballnet.pt. Cached per model and keeper-fix setting."""
    path = os.path.join(CACHE_DIR, "bar_ath_ballnet_{}_{}.pkl".format(model_hash(), slots.KEEPER_FIX))
    return _cached(path, lambda: frame_features(load(), *bar_ath_arrays())[0])
