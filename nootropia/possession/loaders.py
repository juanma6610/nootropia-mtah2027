"""Puts the upstream modules on sys.path and caches the slow loads: the kloppy tracking
DataFrame and the upstream nearest-player pass.

Everything else is loaded with the upstream code directly -- games, starter/*, evaluation/*.
"""

import os
import pickle
import sys

TEAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = TEAM_DIR  # the upstream repository: the nearest folder above this one that holds games.py
while not os.path.isfile(os.path.join(ROOT, "games.py")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
for path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if path not in sys.path:
        sys.path.insert(0, path)

import baseline  # noqa: E402  (upstream starter/baseline.py)
import load_atd  # noqa: E402  (upstream starter/load_atd.py)
import numpy as np  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]  # frames per second of the tactical video and its tracking (25)
OUTPUTS_DIR = os.path.join(TEAM_DIR, "outputs")
CACHE_DIR = os.path.join(OUTPUTS_DIR, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "atd_{source}.pkl")
NEAREST_CACHE_PATH = os.path.join(OUTPUTS_DIR, "cache", "nearest_tactical.pkl")


def graded_mask(game=GAME):
    """bool[n_video_frames], True on the graded frames: both halves, as games.half_video_blocks.

    Filter statistics and tuning with it; run the pipeline itself on every frame.
    """
    mask = np.zeros(GAMES[game]["n_video_frames"], dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def _cached(path, build):
    """build() once, pickled at path; later calls read the pickle."""
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    result = build()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def _build_frame_df(source):
    dataset = load_atd.load(source=source)
    home, away = dataset.metadata.teams
    return (dataset.to_df(), [p.player_id for p in home.players],
            [p.player_id for p in away.players])


def load_frame_df(source="tactical"):
    """(frame_df, home_ids, away_ids). frame_df is kloppy's dataset.to_df() for the full match,
    one row per video frame of that source (row i = frame_id i + 1).

    Loading through kloppy takes over a minute, so the result is pickled once and reused.
    """
    return _cached(CACHE_PATH.format(source=source), lambda: _build_frame_df(source))


def nearest_codes():
    """int8[n_video_frames]: upstream baseline.build(), the nearest player's team per frame
    (baseline.HOME / AWAY, NOBODY where the ball is not located). Cached like load_frame_df()."""
    return _cached(NEAREST_CACHE_PATH, baseline.build)
