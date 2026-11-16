"""Training of possession/ballnet.py: targets from the clean DFL source (the ball, read from the raw positions XML;
team possession from DFL's BallPossession flag; the possessor = the player nearest the clean ball within POSSESSOR_M),
each window at random as built or with off-screen players frozen (freeze.py), plus team swap and pitch flips."""

import glob
import os
import re

import numpy as np
import torch
from torch import nn

from . import benchmark, common
from .common import CACHE_DIR, _game_info
from .freeze import freeze
from possession.ballnet import CFG, STEP, THREADS, BallNet, tokens
from possession.loaders import _cached

# Params
POSSESSOR_M = 1.5        # metres: the player nearest the clean ball within this is "on the ball"


def targets(tok, valid, ball, labels, pitch, step=STEP):
    """(ball float32[T, 2] normalised, NaN = none; team float32[T] 1 home / 0 away / NaN; ptr int[T],
    -1 = none). ball: clean metres float[N, 2]; labels: str[N]."""
    length, width = pitch
    b = ball[::step]
    lab = np.asarray(labels)[::step]
    bn = np.stack([b[:, 0] / (length / 2) - 1, b[:, 1] / (width / 2) - 1], -1).astype(np.float32)
    team = np.where(lab == "H", 1.0, np.where(lab == "A", 0.0, np.nan)).astype(np.float32)
    dx = (tok[:, :, 0] - bn[:, None, 0]) * (length / 2)
    dy = (tok[:, :, 1] - bn[:, None, 1]) * (width / 2)
    d = np.where(valid, np.hypot(dx, dy), np.inf)
    d[~np.isfinite(d)] = np.inf
    best = np.argmin(d, 1)
    ptr = np.where((d[np.arange(len(d)), best] < POSSESSOR_M) & np.isin(lab, ["H", "A"]), best, -1)
    bn[~np.isin(lab, ["H", "A", "D"])] = np.nan  # X: not graded
    return bn, team, ptr


def _augment(tok, ball, team, rng):
    """Random team swap and pitch flips of one window (copies)."""
    tok, ball, team = tok.copy(), ball.copy(), team.copy()
    if rng.random() < 0.5:
        tok[..., 5] *= -1
        team = np.where(np.isnan(team), np.nan, 1 - team)
    if rng.random() < 0.5:
        tok[..., 0] *= -1
        tok[..., 2] *= -1
        ball[..., 0] *= -1
    if rng.random() < 0.5:
        tok[..., 1] *= -1
        tok[..., 3] *= -1
        ball[..., 1] *= -1
    return tok, ball, team


def _pad(arr, k, fill=0):
    if arr.shape[1] == k:
        return arr
    return np.concatenate([arr, np.full((arr.shape[0], k - arr.shape[1]) + arr.shape[2:], fill, arr.dtype)], 1)


def _windows(seqs, cfg):
    out = []
    w, s = cfg["window"], cfg["stride"]
    for tok, valid, ball, team, ptr, alt in seqs:
        for a in range(0, len(tok) - w + 1, s):
            if np.isfinite(ball[a:a + w, 0]).any():
                out.append((tok[a:a + w], valid[a:a + w], ball[a:a + w], team[a:a + w], ptr[a:a + w],
                            None if alt is None else alt[a:a + w]))
    return out


def fit(seqs, cfg=None, log=print):
    """BallNet trained on [(tok, valid, ball, team, ptr, frozen_tok or None)] whole-copy sequences: each
    window takes the frozen tokens with probability 1/2 when given."""
    cfg = dict(CFG, **(cfg or {}))
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(THREADS)
    rng = np.random.default_rng(cfg["seed"])
    k = max(s[0].shape[1] for s in seqs)
    seqs = [(_pad(t, k), _pad(v, k, False), b, tm, p, None if alt is None else _pad(alt, k))
            for t, v, b, tm, p, alt in seqs]
    win = _windows(seqs, cfg)
    net = BallNet(cfg["d"], cfg["heads"], cfg["hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(len(win))
        tot = np.zeros(3)
        for a in range(0, len(order), cfg["batch"]):
            batch = [win[i] for i in order[a:a + cfg["batch"]]]
            aug = [_augment(alt if alt is not None and rng.random() < 0.5 else t, b, tm, rng)
                   for t, _, b, tm, _, alt in batch]
            tok = torch.from_numpy(np.stack([x[0] for x in aug]))
            valid = torch.from_numpy(np.stack([x[1] for x in batch]))
            ball = torch.from_numpy(np.stack([x[1] for x in aug]))
            team = torch.from_numpy(np.stack([x[2] for x in aug]))
            ptr = torch.from_numpy(np.stack([x[4] for x in batch]).astype(np.int64))
            t_logit, b_hat, p_logit = net(tok, valid)
            bm = torch.isfinite(ball[..., 0])
            tm = torch.isfinite(team)
            l_ball = nn.functional.smooth_l1_loss(b_hat[bm], ball[bm], beta=0.05) if bm.any() else 0 * b_hat.sum()
            l_team = (bce(t_logit, torch.nan_to_num(team)) * tm).sum() / tm.sum().clamp(min=1)
            l_ptr = ce(p_logit.reshape(-1, p_logit.shape[-1]), ptr.reshape(-1)) if (ptr >= 0).any() else 0 * p_logit.sum()
            loss = cfg["w_ball"] * l_ball + cfg["w_team"] * l_team + cfg["w_ptr"] * l_ptr
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += [float(l_ball), float(l_team), float(l_ptr)]
        log("    epoch {} ball {:.4f} team {:.4f} ptr {:.4f}".format(
            epoch + 1, *(tot / max(1, (len(order) + cfg["batch"] - 1) // cfg["batch"]))))
    return net


def save(net, path):
    torch.save(net.state_dict(), path)


_SET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)" Z="([-\d.]+)"[^>]*BallStatus="(\d)"')
_HALVES = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}


def _read_raw_ball(positions_xml):
    """(half int8[n], xyz float32[n, 3]) of the DFL ball, rows in (half, frame) order, streamed from the
    positions XML (kloppy needs ~4 GB per match)."""
    rows, half, is_ball = [], 0, False
    with open(positions_xml) as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                m = _SET.match(line)
                is_ball, half = m.group(2) == "BALL", _HALVES.get(m.group(1), 9)
            elif is_ball and line.startswith("<Frame "):
                m = _FRAME.match(line)
                rows.append((half, int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    a = np.array(rows)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    return a[:, 0].astype(np.int8), a[:, 2:5].astype(np.float32)


def clean_ball(game, obs, raw_dir=None):
    """float[N, 2] clean DFL ball in a copy's metres, aligned to its frames (row i = clean frame i, as
    make_benchmark.build writes them). DFL's centred X, Y are mapped per half with the signs that best
    match the copy's seen ball (obs). raw_dir: the DFL files (default common.DFL_DIR, read at call time). Cached."""
    raw_dir = raw_dir or common.DFL_DIR
    mid = game.split("-", 1)[1]

    def build():
        pos = sorted(p for p in glob.glob(os.path.join(raw_dir, "*positions_raw_observed*_DFL-MAT-{}.xml".format(mid))))
        if not pos:
            raise FileNotFoundError("DFL positions XML of {} not in {}".format(game, raw_dir))
        return _read_raw_ball(pos[0])
    half, xyz = _cached(os.path.join(CACHE_DIR, "ballnet_clean_{}.pkl".format(game)), build)
    assert len(xyz) == len(obs), (game, len(xyz), len(obs))
    xyz = xyz.astype(float)
    out = np.full((len(xyz), 2), np.nan)
    for h in np.unique(half):
        m = half == h
        seen = m & np.isfinite(obs[:, 0])
        best = None
        for sx in (1, -1):
            for sy in (1, -1):
                cand = np.stack([52.5 + sx * xyz[:, 0], 34 + sy * xyz[:, 1]], -1)
                err = np.median(np.hypot(*(cand[seen] - obs[seen]).T))
                if best is None or err < best[0]:
                    best = (err, cand)
        assert best[0] < 1.0, (game, h, best[0])
        out[m] = best[1][m]
    return out


def training_sequence(game, bench_dir):
    """(tok, valid, ball, team, ptr, frozen_tok) of one copy, for fit(). Cached per build."""
    _, build = _game_info(game, bench_dir)

    def build_():
        x, y, teams, obs, pitch = benchmark.copy_arrays(game, bench_dir)
        labels = np.asarray(benchmark.ownership_table(game, bench_dir)[1]).astype("<U1")
        tok, valid = tokens(x, y, teams, pitch)
        ball, team, ptr = targets(tok, valid, clean_ball(game, obs), labels, pitch)
        # the frozen copy from float64 positions, stored in float32, as the shipped nets were trained
        fx, fy, _ = freeze(*benchmark.copy_positions64(game, bench_dir), obs[:, 0], pitch)
        frozen, fvalid = tokens(fx.astype(np.float32).astype(float), fy.astype(np.float32).astype(float), teams, pitch)
        assert (fvalid == valid).all()
        return tok, valid, ball, team, ptr, frozen
    return _cached(os.path.join(CACHE_DIR, "ballnet_seq_{}_{}.pkl".format(game, build)), build_)
