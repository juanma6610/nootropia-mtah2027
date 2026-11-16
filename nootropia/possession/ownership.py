"""Ownership: who controls the ball while it is alive, as P(home) per frame.

p_home() is the BiGRU of sequence_model.py on two sets of inputs: features() below and ballnet's columns
(ballnet.py). features() is one row of tracking features per frame and works with anonymous track slots: for each
team, its two players nearest the ball and team aggregates, never a player identity. Every team feature has an H_/A_
pair, so training doubles the data by swapping the teams (and the label), and prediction averages a frame with its
swap (swap()): the model cannot learn a home bias. Two inputs are rules kept as evidence: the gated nearest player
(gated()) and the last touch (TOUCH_*).
"""

import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd

from .loaders import CACHE_DIR, _cached  # first: puts the upstream modules on sys.path
from . import slots
from games import GAME, half_video_blocks  # noqa: E402
import baseline  # noqa: E402  (upstream starter/baseline.py)
import possession_metrics as pm  # noqa: E402  (upstream evaluation/possession_metrics.py)

# Params
INPUTS_VERSION = 2       # of features(): part of bar_ath_table()'s cache key
LAG = 2                  # frames each side for velocities: 0.16 s
MAX_PLAYER_SPEED = 12.0  # m/s: faster is a slot jumping to another player
MAX_BALL_SPEED = 40.0    # m/s: a faster ball "speed" is the tracker jumping, not a ball in flight
MOVING = 0.5             # m/s: slower has no direction worth comparing
NEAR, AROUND = 3.0, 8.0  # metres: "at the ball" and "around the ball"
TEAMS = (("H", baseline.HOME), ("A", baseline.AWAY))
RADIUS_M = 2.0               # the gated nearest player is evidence only within this distance of the ball (None: any)...
GATED_MAX_BALL_SPEED = 7.5   # ...and only while the ball is slower than this, m/s: faster is a pass or shot in flight,
                             # nobody's (None: no limit)
TOUCH_RADIUS = 2.0      # metres: a touch is the ball deflecting or speeding up next to a team's player...
TOUCH_TURN_DEG = 30.0   # ...turning by more than this
TOUCH_ACCEL = 20.0      # ...or accelerating by more than this, m/s2 (~25-30 candidates per live minute)


def gated(nearest, distance, ball_speed, runs, radius=None, max_speed=None):
    """float[N] p_home from the nearest player's team: 1.0 / 0.0 where it counts as evidence, held
    through every frame where it does not (ball not located, too far, or in flight) inside each run.

    nearest: baseline.HOME / AWAY / NOBODY per frame; distance: metres from the ball to that player;
    ball_speed: m/s (features.ball_speed); runs: [(first, stop)].
    """
    evidence = nearest != baseline.NOBODY
    if radius is not None:
        evidence &= distance <= radius
    if max_speed is not None:
        evidence &= ~((ball_speed > max_speed) & (ball_speed <= MAX_BALL_SPEED))
    raw = np.where(evidence, nearest, pm.X).astype(np.int8)
    held = pm.fill_abstentions(raw, runs)
    return np.where(held == pm.H, 1.0, np.where(held == pm.A, 0.0, np.nan))


def _nanmean(values, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def features(x, y, teams, ball, half, fps, runs):
    """DataFrame of model inputs, one row per frame.

    x, y: float[N, S] slot positions in metres (NaN = not tracked); teams: int8[N, S] team per slot
    and frame (baseline.HOME / AWAY, -1 ignore); ball: float[N, 2] metres (NaN = not located);
    half: int[N] 1 / 2 / 0; runs: [(first, stop)] graded runs, for the held rule.
    """
    n = len(x)
    rows = np.arange(n)
    vx, vy = slots.velocity(x, fps, LAG), slots.velocity(y, fps, LAG)
    pspeed = np.hypot(vx, vy)
    bad = ~(pspeed <= MAX_PLAYER_SPEED)
    vx[bad], vy[bad], pspeed[bad] = np.nan, np.nan, np.nan

    bv = slots.velocity(ball, fps, LAG)
    bspeed = np.hypot(bv[:, 0], bv[:, 1])
    jump = ~(bspeed <= MAX_BALL_SPEED)
    bv[jump], bspeed[jump] = np.nan, np.nan
    before, after = np.roll(bv, LAG, axis=0), np.roll(bv, -LAG, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_turn = (before * after).sum(1) / (np.hypot(*before.T) * np.hypot(*after.T))
    accel = np.hypot(*(after - before).T) / (2 * LAG / fps)

    dx, dy = x - ball[:, :1], y - ball[:, 1:]
    dist = np.hypot(dx, dy)
    out = {"ball_seen": np.isfinite(ball[:, 0]).astype(float), "ball_speed": bspeed,
           "ball_turn": np.clip(cos_turn, -1, 1), "ball_accel": accel}

    metadata = np.where((teams == baseline.HOME).sum(0) >= (teams == baseline.AWAY).sum(0),
                        baseline.HOME, baseline.AWAY)
    left = slots.home_defends_left(x, metadata, half)
    home_attack = np.zeros(n)  # +1 where home attacks towards x = length
    for number, defends_left in left.items():
        home_attack[half == number] = 1.0 if defends_left else -1.0

    for name, code in TEAMS:
        d = np.where(teams == code, dist, np.inf)
        d[~np.isfinite(d)] = np.inf
        order = np.argsort(d, axis=1)[:, :2]
        for k in range(2):
            idx = order[:, k]
            dk = d[rows, idx]
            ok = np.isfinite(dk)
            rel_vx, rel_vy = bv[:, 0] - vx[rows, idx], bv[:, 1] - vy[rows, idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                # minus d/dt of the ball-player distance: > 0 while they close in
                closing = (dx[rows, idx] * rel_vx + dy[rows, idx] * rel_vy) / dk
                align = (vx[rows, idx] * bv[:, 0] + vy[rows, idx] * bv[:, 1]) / (pspeed[rows, idx] * bspeed)
            align[~((pspeed[rows, idx] > MOVING) & (bspeed > MOVING))] = np.nan
            out["{}_d{}".format(name, k + 1)] = np.where(ok, dk, np.nan)
            out["{}_closing{}".format(name, k + 1)] = np.where(ok, closing, np.nan)
            out["{}_relspeed{}".format(name, k + 1)] = np.where(ok, np.hypot(rel_vx, rel_vy), np.nan)
            out["{}_align{}".format(name, k + 1)] = np.where(ok, align, np.nan)
            out["{}_speed{}".format(name, k + 1)] = np.where(ok, pspeed[rows, idx], np.nan)
        mine = teams == code
        out[name + "_n_near"] = (mine & (dist < NEAR)).sum(1).astype(float)
        out[name + "_n_around"] = (mine & (dist < AROUND)).sum(1).astype(float)
        out[name + "_motion_near"] = _nanmean(np.where(mine & (dist < AROUND), pspeed, np.nan))
        sign = home_attack if code == baseline.HOME else -home_attack
        out[name + "_advance"] = _nanmean(np.where(mine, vx, np.nan)) * sign
        out[name + "_speed_team"] = _nanmean(np.where(mine, pspeed, np.nan))

    # the current rule, as evidence: gated nearest player now, and held through gaps
    masked = np.where(np.isfinite(dist) & (teams >= 0), dist, np.inf)
    best = np.argmin(masked, axis=1)
    nearest = np.where(np.isfinite(masked[rows, best]), teams[rows, best], baseline.NOBODY)
    near_d = np.fmin(out["H_d1"], out["A_d1"])
    held = gated(nearest, near_d, bspeed, runs, RADIUS_M, GATED_MAX_BALL_SPEED)
    now = gated(nearest, near_d, bspeed, [], RADIUS_M, GATED_MAX_BALL_SPEED)
    out["rule_held"] = held
    out["rule_now"] = np.where(np.isnan(now), 0.0, 2 * now - 1)
    seen = np.isfinite(ball[:, 0])
    since = np.arange(n) - np.maximum.accumulate(np.where(seen, np.arange(n), 0))
    out["ball_since_seen_s"] = np.log1p(since / fps)

    # last touch: the ball turning or accelerating sharply next to a team's nearest player, with no
    # opponent nearer (TOUCH_*), held until the next touch by anyone
    turn = np.degrees(np.arccos(np.clip(out["ball_turn"], -1, 1)))
    sharp = seen & ((turn > TOUCH_TURN_DEG) | (out["ball_accel"] > TOUCH_ACCEL))
    home_d, away_d = out["H_d1"], out["A_d1"]
    touch = np.where(sharp & (home_d <= TOUCH_RADIUS) & ~(away_d < home_d), 1.0,
                     np.where(sharp & (away_d <= TOUCH_RADIUS) & ~(home_d < away_d), -1.0, 0.0))
    touched = touch != 0
    last = np.maximum.accumulate(np.where(touched, np.arange(n), -1))
    out["touch_now"] = touch
    out["last_touch"] = np.where(last >= 0, touch[np.maximum(last, 0)], 0.0)
    out["since_touch_s"] = np.where(last >= 0, np.log1p((np.arange(n) - last) / fps), np.nan)
    return pd.DataFrame(out)


def swap(table):
    """The same frames with home and away exchanged (inputs only)."""
    out = table.copy()
    for column in table.columns:
        if column.startswith("H_"):
            out[column], out["A_" + column[2:]] = table["A_" + column[2:]], table[column]
    out["rule_held"] = 1 - table["rule_held"]
    for column in ("rule_now", "touch_now", "last_touch"):
        if column in table:
            out[column] = -table[column]
    return out


def bar_ath_table():
    """features() for BAR-ATH, with the keeper slots corrected (slots.slot_teams). Cached."""
    from .loaders import load_frame_df
    from games import GAME, GAMES, half_video_blocks  # noqa: E402

    def build():
        frame_df, home_ids, away_ids = load_frame_df()
        ids = list(home_ids) + list(away_ids)
        x, y = slots.slot_positions(frame_df, ids)
        teams = slots.slot_teams(frame_df, home_ids, away_ids, fix=slots.KEEPER_FIX)
        length, width = GAMES[GAME]["pitch"]
        ball = frame_df[["ball_x", "ball_y"]].to_numpy(dtype=float) * (length, width)
        runs = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
        table = features(x, y, teams, ball, slots.half_of_frame(len(frame_df)), GAMES[GAME]["fps"], runs)
        floats = table.select_dtypes("float64").columns
        table[floats] = table[floats].astype(np.float32)
        return table
    settings = json.dumps([INPUTS_VERSION, RADIUS_M, GATED_MAX_BALL_SPEED, slots.KEEPER_FIX])
    key = hashlib.sha1(settings.encode()).hexdigest()[:8]
    return _cached(os.path.join(CACHE_DIR, "bar_ath_own_v{}_{}.pkl".format(INPUTS_VERSION, key)), build)


def bar_ath_inputs():
    """The ownership BiGRU's inputs for BAR-ATH: bar_ath_table() plus ballnet's columns."""
    from . import ballnet
    return pd.concat([bar_ath_table().reset_index(drop=True), ballnet.bar_ath_columns()], axis=1)


def p_home():
    """float[N]: probability that home owns the ball, assuming it is alive: the BiGRU
    (models/sequence.pt) on BAR-ATH's ownership inputs and ballnet's (models/ballnet.pt), each half
    predicted on its own."""
    from . import sequence_model  # here, not at the top: it imports this module
    runs = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    net, scaler, config = sequence_model.load()
    return sequence_model.predict(net, scaler, bar_ath_inputs(), runs, config)
