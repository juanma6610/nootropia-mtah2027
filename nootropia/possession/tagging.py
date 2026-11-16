"""Smart Tagging: per-frame priors from the automatic phases and set pieces.

How much a Team tag is worth depends on the code. A phase is 12-35 s long and covers more than one
possession, so the question per code is: on what share of its frames does the tagged team own the
ball? There are too few tags in the public window to answer that (1-5 per code), so reliability()
answers it over the whole match against the tracking instead, without ground truth:

  agree   share of a code's frames where the nearest player (upstream baseline) is on the tagged
          team, counting only frames where the ball is seen, the match is graded and players are
          moving (speed_2s > ALIVE_SPEED, to leave stoppages out)
  p       the same share against the true owner. The nearest player is itself right only
          NEAREST_ACCURACY of the time, so agree = p q + (1 - p)(1 - q), i.e.
          p = (agree - (1 - q)) / (2q - 1), clipped to [0.5, MAX_RELIABILITY]

Set pieces mark restarts, but a SET PIECES tag starts at the set-up, 4-8 s before the ball is
played, and runs on ~30 s into live play. restarts() finds the actual restart after each tag start
in the tracking: the first point where the players' 1 s mean speed rises RESTART_RISE above its
lowest value since the tag start, minus RESTART_LAG.
"""

import numpy as np
import pandas as pd

from .features import load_features
from .loaders import FPS, graded_mask, nearest_codes
from games import GAME, GAMES  # noqa: E402
import baseline  # noqa: E402  (upstream starter/baseline.py)
import load_smart_tagging  # noqa: E402  (upstream starter/load_smart_tagging.py)

# Params
DEFENDING_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
NO_TEAM_CODES = {"GOALS"}       # tagged with the scorer, but the interval is mostly celebration
TAG_TEAMS = {"barcelona": 1.0, "atletic club": 0.0}  # Team tag -> home
NEAREST_ACCURACY = 0.78         # nearest player's team = GT team, ball seen and alive, public window (notebook 02)
ALIVE_SPEED = 1.6               # m/s, speed_2s: ~2.2 in play, ~1.2 in stoppages (features.py)
MAX_RELIABILITY = 0.95          # never trust one automatic tag more than this
RESTART_SEARCH = 12 * FPS       # frames after a set-piece tag start to look for the restart
RESTART_RISE = 0.5              # m/s rise of speed_1s above its minimum since the tag start
RESTART_LAG = 0                 # frames to step back from the rise (benchmark v3: none)
RESTART_FALLBACK = 5 * FPS      # frames after the tag start when no rise is found
DEAD_LEAD = 2 * FPS             # frames before a tag start asserted dead: the stoppage began earlier,
                                # but how much earlier is for the other cues to say
ALIVE_AFTER = 2 * FPS           # frames after a restart asserted alive


def instances(game=GAME):
    """DataFrame, one row per tag: code (upper case), team (1.0 home / 0.0 away / NaN), start, end
    (video frames, half-open, clipped to the video), type (set-piece Type tag or "")."""
    n = GAMES[game]["n_video_frames"]
    rows = []
    for tag in load_smart_tagging.load(game):
        rows.append({"code": tag["code"].strip().upper(),
                     "team": TAG_TEAMS.get((tag["tags"].get("Team") or "").strip().lower(), np.nan),
                     "start": max(tag["start_frame"], 0), "end": min(tag["end_frame"], n),
                     "type": tag["tags"].get("Type") or ""})
    return pd.DataFrame(rows)


def reliability(game=GAME):
    """DataFrame per voting code: n tags, n with a team, proxy frames, agree and p (module docstring)."""
    near = nearest_codes()
    proxy = (near != baseline.NOBODY) & graded_mask(game) \
        & (load_features()["speed_2s"].to_numpy() > ALIVE_SPEED)
    near_home = near == baseline.HOME

    tags = instances(game)
    tags = tags[~tags.code.isin(DEFENDING_CODES | NO_TEAM_CODES)]
    rows = []
    for code, group in tags.groupby("code"):
        agree_frames = total = 0
        for tag in group.dropna(subset=["team"]).itertuples():
            frames = slice(tag.start, tag.end)
            on = proxy[frames]
            agree_frames += int((near_home[frames][on] == bool(tag.team)).sum())
            total += int(on.sum())
        agree = agree_frames / total if total else np.nan
        q = NEAREST_ACCURACY
        p = np.clip((agree - (1 - q)) / (2 * q - 1), 0.5, MAX_RELIABILITY) if total else 0.5
        rows.append({"code": code, "n": len(group), "n_team": int(group.team.notna().sum()),
                     "proxy_frames": total, "agree": agree, "p": p})
    return pd.DataFrame(rows).set_index("code").sort_values("p", ascending=False)


def tag_team(game=GAME):
    """float[N]: P(home) from the Team tags, NaN where no voting tag with a team covers the frame.

    Each frame takes the most reliable covering tag: p if it says home, 1 - p if away. Taking the
    best one rather than multiplying them all keeps it honest, because tags covering the same
    frame are not independent detections (Ball in the Box sits inside Ball in Final Third).
    """
    n = GAMES[game]["n_video_frames"]
    best_p, home = np.full(n, 0.5), np.full(n, np.nan)
    p_of = reliability(game)["p"]
    for tag in instances(game).dropna(subset=["team"]).itertuples():
        if tag.code not in p_of.index:
            continue
        frames = np.arange(tag.start, tag.end)
        better = p_of[tag.code] > best_p[frames]
        best_p[frames[better]] = p_of[tag.code]
        home[frames[better]] = tag.team
    return np.where(np.isnan(home), np.nan, np.where(home == 1.0, best_p, 1.0 - best_p))


def find_restart(speed_1s, start, search=RESTART_SEARCH):
    """Frame of the restart after a set-piece tag starting at `start`, or None (module docstring)."""
    lowest = np.inf
    for frame in range(start, min(start + search, len(speed_1s))):
        if np.isnan(speed_1s[frame]):
            continue
        lowest = min(lowest, speed_1s[frame])
        if speed_1s[frame] - lowest > RESTART_RISE:
            return max(frame - RESTART_LAG, start)
    return None


def restarts(game=GAME):
    """DataFrame, one row per set-piece tag: start (tag start), type, team, restart (frame of the
    first live frame), found (False where RESTART_FALLBACK was used)."""
    speed_1s = load_features()["speed_1s"].to_numpy()
    rows = []
    for tag in instances(game).query("code == 'SET PIECES'").itertuples():
        restart = find_restart(speed_1s, tag.start)
        rows.append({"start": tag.start, "type": tag.type, "team": tag.team,
                     "restart": restart if restart is not None else tag.start + RESTART_FALLBACK,
                     "found": restart is not None})
    return pd.DataFrame(rows)


def tag_dead(game=GAME):
    """float[N]: 1.0 where the tags say the ball is dead, 0.0 where they say it is alive, NaN where
    they say nothing.

    - each set piece: dead from DEAD_LEAD before the tag start to the restart (restarts()), alive for
      ALIVE_AFTER from the restart
    - each goal: dead from the first frame of the GOALS tag where the tracked ball is past a line
      (the tag start if never) to the restart of the next kick-off after a goal

    Says nothing about the stoppages the tagging misses (about half), so NaN is not "alive".
    """
    n = GAMES[game]["n_video_frames"]
    out = np.full(n, np.nan)
    table = restarts(game)
    for row in table.itertuples():
        out[row.restart:min(row.restart + ALIVE_AFTER, n)] = 0.0
    for row in table.itertuples():
        out[max(row.start - DEAD_LEAD, 0):row.restart] = 1.0

    ball_out = load_features()["ball_out_m"].to_numpy()
    kickoffs = table[table.type.str.lower().str.startswith("kick off (after")]
    for goal in instances(game).query("code == 'GOALS'").itertuples():
        after = kickoffs[kickoffs.start > goal.start]
        if after.empty:
            continue
        past_line = np.flatnonzero(ball_out[goal.start:after.start.iloc[0]] > 0)
        first = goal.start + (past_line[0] if len(past_line) else 0)
        out[first:after.restart.iloc[0]] = 1.0
    return out
