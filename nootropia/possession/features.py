"""The shared per-frame feature table: one row per tactical video frame (199319, index = submission
row), built once and cached, so every module reads the same numbers.

    from possession.features import load_features
    feats = load_features()                  # pandas DataFrame, ~10 s from the cache

Columns (distances in metres, speeds in m/s, NaN = not observed):

| group | column | meaning |
|---|---|---|
| ball | `ball_x`, `ball_y` | ATD ball position in metres (0..105.3, 0..68) |
| | `ball_seen` | ATD has a ball on this frame |
| | `ball_speed` | from positions over +/- BALL_SPEED_LAG frames, NaN unless both ends are seen |
| | `ball_speed_1s` | `ball_speed` with tracker jumps (> MAX_BALL_SPEED) removed, centred 1 s average |
| | `ball_out_m` | signed distance outside the pitch lines: > 0 outside, < 0 inside |
| | `ball_out_held` | `ball_out_m` where the ball was last seen, held through ball gaps: a ball lost near or over a line |
| | `ball_since_seen`, `ball_to_seen` | frames since the ball was last seen / until it is next seen (0 when seen) |
| movement | `speed_1s` | mean speed of all tracked players, centred 1 s average: tag restarts are found where it rises |
| | `speed_2s`, `coherence_2s` | mean of both teams' speed / coherence (how much a team's moving players run the same way: length of their mean unit velocity, NaN if fewer than MIN_MOVING move), centred MOVEMENT_WINDOW average. Both drop in stoppages (players stop and drift) |
| keeper | `keeper_ball_m` | distance from the ball to the keeper at its end (keeper_features: the player nearest that goal's centre), NaN when the ball is unseen |
| | `keeper_ball_held` | `keeper_ball_m` where the ball was last seen, held through ball gaps: a ball lost in the keeper's hands |
| | `ball_in_box_held` | the ball was last seen inside a penalty area (1/0) |
| | `keeper_speed_1s` | that keeper's ATD speed, centred 1 s average |

The ball, movement and keeper groups (tracking_features()) exist for any EPTS match: training/ builds them for the benchmark
games too.
"""

import os
import warnings

import numpy as np
import pandas as pd

from .loaders import CACHE_DIR, _cached, load_frame_df
from . import slots
from games import GAME, GAMES  # noqa: E402

# Params
FEATURES_VERSION = 8
FEATURES_CACHE_PATH = os.path.join(CACHE_DIR, "features_v{}.pkl".format(FEATURES_VERSION))
BALL_SPEED_LAG = 2      # frames each side: 0.16 s baseline, enough to average out ~0.25 m jitter
MAX_BALL_SPEED = 40.0   # m/s: faster is the tracker jumping to another object (BAR-ATH p99 is 25)
BALL_SPEED_WINDOW = 25  # frames, centred: 1 s
MOVING_SPEED = 0.5      # m/s: slower players have no direction worth counting
MAX_PLAYER_SPEED = 12.0 # m/s: a faster step between frames is the tracker moving the slot, not a run
MIN_MOVING = 3          # moving players needed for a team coherence
MOVEMENT_WINDOW = 50    # frames, centred: 2 s. Per frame the cue is weak; sustained it separates
                        # dead ball well (scored on the public window and two benchmark matches)


def _nanreduce(fn, values, axis=1):
    """fn (np.nanmin, np.nanmean...) without the all-NaN RuntimeWarning; all-NaN rows give NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(values, axis=axis)


def _frames_since(seen):
    """int: frames since seen was last True (0 where True), NaN before the first True."""
    index = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, index, -1))
    return np.where(last >= 0, index - last, np.nan)


def ball_features(frame_df, pitch, fps):
    length, width = pitch
    xy = frame_df[["ball_x", "ball_y"]].to_numpy(dtype=float) * (length, width)
    seen = np.isfinite(xy).all(axis=1)

    lag = BALL_SPEED_LAG
    speed = np.full(len(xy), np.nan)
    speed[lag:-lag] = np.linalg.norm(xy[2 * lag:] - xy[:-2 * lag], axis=1) \
        / (2 * lag / fps)
    speed_1s = pd.Series(np.where(speed <= MAX_BALL_SPEED, speed, np.nan)).rolling(
        BALL_SPEED_WINDOW + 1, center=True, min_periods=BALL_SPEED_WINDOW // 5).mean().to_numpy()

    x, y = xy[:, 0], xy[:, 1]
    out = np.max([-x, x - length, -y, y - width], axis=0)
    return {"ball_x": x, "ball_y": y, "ball_seen": seen, "ball_speed": speed,
            "ball_speed_1s": speed_1s, "ball_out_m": out,
            "ball_out_held": pd.Series(out).ffill().to_numpy(),
            "ball_since_seen": _frames_since(seen),
            "ball_to_seen": _frames_since(seen[::-1])[::-1]}


def movement_features(frame_df, home_ids, away_ids, pitch, fps):
    length, width = pitch
    out, speeds, coherence = {}, [], []
    for ids in (home_ids, away_ids):
        px = frame_df[["{}_x".format(p) for p in ids]].to_numpy(dtype=float) * length
        py = frame_df[["{}_y".format(p) for p in ids]].to_numpy(dtype=float) * width
        vx = np.vstack([np.full((1, len(ids)), np.nan), np.diff(px, axis=0) * fps])
        vy = np.vstack([np.full((1, len(ids)), np.nan), np.diff(py, axis=0) * fps])
        speed = np.hypot(vx, vy)
        moving = (speed > MOVING_SPEED) & (speed < MAX_PLAYER_SPEED)
        with np.errstate(invalid="ignore", divide="ignore"):
            ux = np.where(moving, vx / speed, 0.0).sum(axis=1) / moving.sum(axis=1)
            uy = np.where(moving, vy / speed, 0.0).sum(axis=1) / moving.sum(axis=1)
        coherence.append(np.where(moving.sum(axis=1) >= MIN_MOVING, np.hypot(ux, uy), np.nan))
        speeds.append(frame_df[["{}_s".format(p) for p in ids]].to_numpy(dtype=float))

    def sustained(values, window=MOVEMENT_WINDOW):
        return pd.Series(values).rolling(window + 1, center=True,
                                         min_periods=window // 5).mean().to_numpy()

    speed = _nanreduce(np.nanmean, np.hstack(speeds))
    out["speed_1s"] = sustained(speed, fps)
    out["speed_2s"] = sustained(speed)
    out["coherence_2s"] = sustained(_nanreduce(
        np.nanmean, np.column_stack(coherence)))
    return out


def keeper_features(frame_df, home_ids, away_ids, pitch, fps, teams=None):
    """The keeper at the end where the ball was last seen: the tracked player of either team nearest the
    centre of that goal, per frame. teams: int8[N, n_slots] per-frame slot teams (slots.slot_teams; -1 = ignore the
    slot), None = every slot counts."""
    length, width = pitch
    ball = frame_df[["ball_x", "ball_y"]].to_numpy(dtype=float) * (length, width)
    held = pd.DataFrame(ball).ffill().to_numpy()  # where the ball was last seen
    ids = list(home_ids) + list(away_ids)
    x = frame_df[["{}_x".format(p) for p in ids]].to_numpy(dtype=float) * length
    y = frame_df[["{}_y".format(p) for p in ids]].to_numpy(dtype=float) * width
    s = frame_df[["{}_s".format(p) for p in ids]].to_numpy(dtype=float)
    goal_x = np.where(held[:, 0] < length / 2, 0.0, length)
    d_goal = np.hypot(x - goal_x[:, None], y - width / 2)
    d_goal[~np.isfinite(d_goal)] = np.inf
    if teams is not None:
        d_goal[teams < 0] = np.inf
    k = np.argmin(d_goal, axis=1)
    rows = np.arange(len(k))
    found = np.isfinite(d_goal[rows, k])
    to_ball = np.where(found & np.isfinite(ball[:, 0]), np.hypot(x[rows, k] - ball[:, 0], y[rows, k] - ball[:, 1]), np.nan)
    in_box = np.where(np.isfinite(ball[:, 0]), (np.minimum(ball[:, 0], length - ball[:, 0]) < slots.BOX_DEPTH)
                      & (np.abs(ball[:, 1] - width / 2) < slots.BOX_HALF_WIDTH), np.nan)
    speed = np.where(found, s[rows, k], np.nan)
    return {"keeper_ball_m": to_ball,
            "keeper_ball_held": pd.Series(to_ball).ffill().to_numpy(),
            "ball_in_box_held": pd.Series(in_box).ffill().to_numpy(),
            "keeper_speed_1s": pd.Series(np.where(speed < MAX_PLAYER_SPEED, speed, np.nan)).rolling(
                fps + 1, center=True, min_periods=fps // 5).mean().to_numpy()}


def tracking_features(frame_df, home_ids, away_ids, pitch, fps, teams=None):
    """dict of the tracking-only columns for any EPTS match: kloppy's to_df(), the two teams' player ids (metadata
    teams), pitch (length, width) in metres and the frame rate. teams: per-frame slot teams for keeper_features
    (BAR-ATH: slots.slot_teams), None on the benchmark."""
    columns = ball_features(frame_df, pitch, fps)
    columns.update(movement_features(frame_df, home_ids, away_ids, pitch, fps))
    columns.update(keeper_features(frame_df, home_ids, away_ids, pitch, fps, teams))
    return columns


def build(game=GAME):
    """The feature table, computed from scratch (a few minutes the first time, for the ATD loads)."""
    frame_df, home_ids, away_ids = load_frame_df()
    columns = tracking_features(frame_df, home_ids, away_ids, GAMES[game]["pitch"], GAMES[game]["fps"],
                                slots.slot_teams(frame_df, home_ids, away_ids, game=game))

    table = pd.DataFrame(columns)
    table.index.name = "frame"
    floats = table.select_dtypes("float64").columns
    table[floats] = table[floats].astype(np.float32)
    return table


def load_features():
    """The feature table, cached under outputs/cache (gitignored: it is derived match data)."""
    return _cached(FEATURES_CACHE_PATH, build)
