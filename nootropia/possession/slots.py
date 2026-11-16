"""Which team each track slot is on, frame by frame, and the nearest player's team from that.

The ATD metadata gives each of the 46 slots one team for the whole match. That is right on average
but not frame by frame: a slot is a tracklet, and the tracker reuses it for other players. It
matters most for the keeper slots. On BAR-ATH:

    slot 46 (away keeper)  in a penalty area on 96% of its sightings, but in 9 runs (median 51 s,
                           up to 230 s; 16,392 graded frames) it sits at the HOME goal, 3-15 m from
                           the home keeper (slot 24), while no other away slot is near the away goal
    slot 24 (home keeper)  in a penalty area on 89%, wrong side on 1,610 graded frames

So in those runs the away keeper is lost and slot 46 tracks someone at the home end. On the 26
public frames where it is the nearest player to the ball, the GT says H on all 26.

Rule: a keeper slot (KEEPER_BOX_SHARE of its sightings in a penalty area) seen past halfway is not
its keeper; it is given to the team defending that end ("relabel"), or dropped ("drop").
"""

import numpy as np

from games import GAME, GAMES, half_video_blocks  # noqa: E402
import baseline  # noqa: E402  (upstream starter/baseline.py)

# Params
KEEPER_BOX_SHARE = 0.6   # the two keeper slots sit at 0.89 / 0.96, no other slot above 0.20
BOX_DEPTH = 16.5         # metres, penalty area
BOX_HALF_WIDTH = 20.16   # metres, half of 40.32
KEEPER_FIX = "relabel"


def slot_positions(frame_df, ids, game=GAME):
    """(x, y): float[N, n_slots] in metres, NaN where the slot is not tracked."""
    length, width = GAMES[game]["pitch"]
    x = frame_df[["{}_x".format(p) for p in ids]].to_numpy(dtype=float) * length
    y = frame_df[["{}_y".format(p) for p in ids]].to_numpy(dtype=float) * width
    return x, y


def half_of_frame(n, game=GAME):
    """int8[N]: 1 / 2 inside the graded halves, 0 elsewhere."""
    half = np.zeros(n, dtype=np.int8)
    for number, first, last in half_video_blocks(game):
        half[first:last + 1] = number
    return half


def keeper_slots(x, y, game=GAME):
    """Indices of the slots that spend at least KEEPER_BOX_SHARE of their sightings in a box."""
    length, width = GAMES[game]["pitch"]
    in_box = ((x < BOX_DEPTH) | (x > length - BOX_DEPTH)) & (np.abs(y - width / 2) < BOX_HALF_WIDTH)
    share = in_box.sum(axis=0) / np.maximum(np.isfinite(x).sum(axis=0), 1)
    return np.flatnonzero(share >= KEEPER_BOX_SHARE)


def home_defends_left(x, metadata_teams, half):
    """{half: bool}: does home defend the x = 0 goal in that half. From the team centroids over the
    half, a fact about the match rather than a parameter."""
    out = {}
    for number in np.unique(half[half > 0]):
        rows = x[half == number]
        out[int(number)] = (np.nanmean(rows[:, metadata_teams == baseline.HOME])
                            < np.nanmean(rows[:, metadata_teams == baseline.AWAY]))
    return out


def slot_teams(frame_df, home_ids, away_ids, fix=KEEPER_FIX, game=GAME):
    """int8[N, n_slots]: team of each slot on each frame (baseline.HOME / AWAY), -1 = ignore it.
    Slot order is home_ids + away_ids."""
    ids = list(home_ids) + list(away_ids)
    metadata = np.array([baseline.HOME] * len(home_ids) + [baseline.AWAY] * len(away_ids), np.int8)
    teams = np.tile(metadata, (len(frame_df), 1))
    if fix is None:
        return teams

    length = GAMES[game]["pitch"][0]
    x, y = slot_positions(frame_df, ids, game)
    half = half_of_frame(len(frame_df), game)
    for number, left in home_defends_left(x, metadata, half).items():
        rows = np.flatnonzero(half == number)
        # the team defending the end a frame's slot stands in
        end_team = np.where((x[rows] < length / 2) == left, baseline.HOME, baseline.AWAY)
        for slot in keeper_slots(x, y, game):
            wrong = end_team[:, slot] != metadata[slot]
            teams[rows[wrong], slot] = end_team[wrong, slot] if fix == "relabel" else -1
    return teams


def velocity(p, fps, lag=2, dtype=float):
    """float[N, ...] of dtype: centred finite difference of p over +/- lag frames, NaN at the edges."""
    v = np.full(p.shape, np.nan, dtype=dtype)
    v[lag:-lag] = (p[2 * lag:] - p[:-2 * lag]) / (2 * lag / fps)
    return v
