"""Dead ball: when the ball is out of play, and who restarts it.

p_dead() is a BiGRU (sequence_model.py, target "dead") on the tracking alone: per frame, the stoppage cues of design()
(player speeds and coherence, ball speed, seen, gap length, out of the pitch, the keeper near where the ball was lost)
next to the ownership inputs
(ownership.features(): who is near the ball, how it moves). They exist for any EPTS match, so the model is trained on
the DFL benchmark games (labels from the tracking's own BallStatus flags, the same kind of derivation as the challenge
ground truth; training/train_dead.py) and applied unchanged to BAR-ATH, never trained on it. Its training copies hide
the ball through keeper holds (live) and through most stoppages that start in a box (dead), as BAR-ATH's tracker loses
it there (training/make_benchmark.py), so a ball lost in the keeper's hands is not read as a stoppage.

edges() turns ball events into hints for the decoder and restart_team() names the restarting team.
"""

import os

import numpy as np
import pandas as pd

from .loaders import FPS, TEAM_DIR, nearest_codes  # first: puts the upstream modules on sys.path
import baseline  # noqa: E402  (upstream starter/baseline.py)

# Params
MODEL_PATH = os.path.join(TEAM_DIR, "models", "dead_sequence.pt")
CONFIG = {"target": "dead"}   # sequence_model.DEFAULT otherwise: 64 x 2, 5 fps, 10 s windows
GAP_CAP_S = 60.0        # seconds: longer ball gaps are all "very long"
OUT_CLIP = (-10.0, 5.0)  # metres: deeper than 10 m inside the pitch says nothing about the line

# Model inputs, built by design(). Counts of tracked objects are left out on purpose: the
# benchmark's simulated tracker does not reproduce the real one's object counts.
# The keeper inputs tell a ball lost in the keeper's hands (live) from one lost at a stoppage: both are a long gap with
# slow players. A saved model reads only the columns it was trained on (Scaler.columns).
INPUTS = ["speed_1s", "speed_2s", "coherence_2s", "ball_speed_1s", "ball_seen", "ball_gap_s",
          "ball_out_held", "ball_out_now", "keeper_ball_held", "ball_in_box_held", "keeper_speed_1s"]
KEEPER_CLIP_M = 20.0    # metres: farther from the keeper says nothing about a hold


NO_RESTART = -1
RESTART_TEAM_WINDOW = 2 * FPS   # frames after a restart to find the ball next to a player
NEAR_LINE_M = 2.0       # a ball lost this close to a line (or beyond it) hints a stoppage start
MIN_LOST = 25           # frames the ball must stay lost for that hint
KICK_STILL = 12         # frames the tracked ball must be near-still before a kick counts
STILL_SPEED, KICK_SPEED = 1.5, 4.0   # m/s
HINT_SPREAD = 12        # frames: a hint fades linearly to 0 this far either side of its cue


def design(table, fps=FPS):
    """DataFrame of INPUTS from a feature table (BAR-ATH's or a benchmark game's). NaN stays NaN:
    the model pipeline imputes it and adds a missing flag."""
    seen = table["ball_seen"].to_numpy(dtype=bool)
    gap = (table["ball_since_seen"].to_numpy(dtype=float) + table["ball_to_seen"].to_numpy(dtype=float)) / fps
    return pd.DataFrame({
        "speed_1s": table["speed_1s"].to_numpy(dtype=float),
        "speed_2s": table["speed_2s"].to_numpy(dtype=float),
        "coherence_2s": table["coherence_2s"].to_numpy(dtype=float),
        "ball_speed_1s": table["ball_speed_1s"].to_numpy(dtype=float),
        "ball_seen": seen.astype(float),
        # length of the gap the frame sits in (0 when seen); NaN gaps run to a match edge
        "ball_gap_s": np.where(seen, 0.0, np.log1p(np.minimum(np.nan_to_num(gap, nan=GAP_CAP_S), GAP_CAP_S))),
        "ball_out_held": np.clip(table["ball_out_held"].to_numpy(dtype=float), *OUT_CLIP),
        "ball_out_now": (np.nan_to_num(table["ball_out_m"].to_numpy(dtype=float), nan=-1.0) > 0).astype(float),
        "keeper_ball_held": np.minimum(table["keeper_ball_held"].to_numpy(dtype=float), KEEPER_CLIP_M),
        "ball_in_box_held": table["ball_in_box_held"].to_numpy(dtype=float),
        "keeper_speed_1s": table["keeper_speed_1s"].to_numpy(dtype=float),
    }, index=table.index)


def sequence_inputs(cues, own):
    """DataFrame, one row per frame: the stoppage cues (design()) next to the ownership inputs `own`
    (ownership.features()), minus the columns both have. Both one row per frame from 0."""
    x = cues.reset_index(drop=True)
    own = own.drop(columns=[c for c in own.columns if c in x.columns]).reset_index(drop=True)
    return pd.concat([x, own], axis=1)


def bar_ath_sequence():
    """BAR-ATH's inputs, every frame."""
    from . import ownership
    from .features import load_features

    return sequence_inputs(design(load_features()), ownership.bar_ath_table())


def p_dead():
    """float[N]: P(dead) per frame for BAR-ATH: the BiGRU (models/dead_sequence.pt) on bar_ath_sequence(), each half
    predicted on its own, with the players-only opinion (players_dead.py) averaged in where the ball is unseen."""
    from . import sequence_model as sm
    from .decode import bar_ath_runs  # here: decode imports this module

    from . import players_dead

    net, scaler, config = sm.load(MODEL_PATH)
    return players_dead.bar_ath(sm.predict(net, scaler, bar_ath_sequence(), bar_ath_runs(), config))


def _spread(frames, n):
    """float[n] in 0..1: 1 at each cue frame, fading linearly to 0 over HINT_SPREAD frames."""
    hint = np.zeros(n)
    hint[np.asarray(frames, dtype=int)] = 1.0
    kernel = 1 - np.abs(np.arange(-HINT_SPREAD, HINT_SPREAD + 1)) / (HINT_SPREAD + 1)
    return np.minimum(np.convolve(hint, kernel, mode="same"), 1.0)


def edges(feats, restarts):
    """(start_hint, end_hint) float[N] in 0..1 for decode.decode(hints=...).

    start: the tracked ball crosses a pitch line, or is lost for MIN_LOST frames within NEAR_LINE_M
           of a line (or beyond it)
    end:   the restarts of the set pieces (restarts: frames, tagging.restarts().restart), and a
           tracked ball kicked (> KICK_SPEED) after KICK_STILL frames below STILL_SPEED
    feats: a feature table with the tracking columns (BAR-ATH's or a benchmark game's).
    """
    n = len(feats)
    out_m = feats["ball_out_m"].to_numpy(dtype=float)
    seen = feats["ball_seen"].to_numpy(dtype=bool)
    since = feats["ball_to_seen"].to_numpy(dtype=float)
    crosses = np.flatnonzero((out_m[1:] > 0) & (out_m[:-1] <= 0)) + 1
    lost = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    lost = lost[(out_m[lost - 1] > -NEAR_LINE_M) & ~(since[lost] < MIN_LOST)]

    speed = feats["ball_speed"].to_numpy(dtype=float)
    still = pd.Series(speed < STILL_SPEED).rolling(KICK_STILL).sum().shift(1).to_numpy() == KICK_STILL
    kicks = np.flatnonzero(still & (speed > KICK_SPEED))
    return (_spread(np.concatenate([crosses, lost]), n),
            _spread(np.concatenate([np.asarray(restarts, dtype=int), kicks]), n))


def restart_team(p_dead, set_pieces=None, nearest=None):
    """int8[N]: baseline.HOME / AWAY at the first alive frame after a stoppage, NO_RESTART elsewhere.

    A stoppage is a run of p_dead >= 0.5. The restarting team is the nearest player's team at the
    first frame within RESTART_TEAM_WINDOW of the restart where the ball is located: whoever takes
    the throw-in, corner or free kick stands on it. Where the ball is not located, the Team tag of a
    set piece restarting within the window (set_pieces: tagging.restarts()), if it has one.
    nearest: the nearest player's team per frame (default: BAR-ATH's, loaders.nearest_codes()).
    """
    nearest = nearest_codes() if nearest is None else nearest
    dead = np.nan_to_num(p_dead, nan=0.0) >= 0.5
    out = np.full(len(p_dead), NO_RESTART, dtype=np.int8)
    for frame in np.flatnonzero(dead[:-1] & ~dead[1:]) + 1:
        window = nearest[frame:frame + RESTART_TEAM_WINDOW]
        located = np.flatnonzero(window != baseline.NOBODY)
        if len(located):
            out[frame] = window[located[0]]
        elif set_pieces is not None:
            tagged = set_pieces[(abs(set_pieces.restart - frame) <= RESTART_TEAM_WINDOW)
                                & set_pieces.team.notna()]
            if len(tagged):
                out[frame] = baseline.HOME if tagged.team.iloc[0] == 1.0 else baseline.AWAY
    return out
