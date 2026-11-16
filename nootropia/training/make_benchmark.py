#!/usr/bin/env python3
"""Build a possession benchmark from public matches, in the same format as the challenge ATD.

For every source match this writes, under --out/<GAME>/:
    <GAME>_tactical_atd.xml   EPTS metadata (teams, anonymous "Track n" slots), kloppy-readable
    <GAME>_tactical_atd.txt   raw tracking, "<frame_id>:<x,y,s>;...:<bx,by,bs>", 0-1 coordinates
    <GAME>_gt.csv             frame,label (H/A/D on graded frames, X elsewhere), video frame = frame_id - 1
    <GAME>_info.json          pitch, graded blocks, label statistics, noise actually applied
and a manifest.json listing all games. Our pipeline and starter/baseline.py read the pair unchanged.

Sources
    metrica:1 metrica:2 metrica:3     Metrica Sports sample data (github.com/metrica-sports/sample-data).
                                      The needed files are downloaded into --metrica-dir automatically;
                                      an existing clone there is used as is.
                                      Labels are DERIVED FROM EVENTS (see metrica_labels).
    dfl:J03WMX ... (7 matches)        DFL / IDSSE open data (Bassek et al. 2025, CC-BY 4.0), downloaded
                                      by kloppy from figshare -- which now blocks scripted downloads;
                                      on failure the error names the two files to fetch in a browser
                                      for dfl-files: below. Labels are the per-frame BallStatus /
                                      BallPossession flags of the tracking itself: the same kind of
                                      derivation as the challenge ground truth.
    dfl-files:<meta.xml>,<positions.xml>   a DFL match you already have on disk.

Noise
    The source tracking is far cleaner than the challenge ATD, so it is degraded before writing:
    ball dropout in bursts (more likely in crowds and on fast balls) until --ball-missing of graded
    frames have no ball; player tracks cut into anonymous tracklets packed into slots; a few
    tracklets given the wrong team; position error. `calibrate` measures the real ATD so these
    numbers can match it.

    Position error is a slowly drifting offset, never independent per-frame jitter: the real ATD
    is smooth (second difference of player positions: median 0.0014 m), and per-frame jitter of
    0.15 m made every movement cue noise (median 0.43 m, speeds inflated, directions random).
    The raw DFL positions are themselves rougher than the ATD (0.010 m), so they are first smoothed
    over *_smooth_frames, which brings players to 0.0013 m. `build` reports the wobble it produced
    (noise_applied) and `calibrate` the ATD's, so the match can be checked on every rebuild.

    The real tracker also loses a ball that leaves the pitch. So the ball is kept only for the first
    ball_out_keep frames of each spell outside the lines.

    DFL freezes the ball's coordinates through every stoppage (zero displacement between all
    adjacent dead-ball frames), which makes "the ball is still" a near-perfect dead-ball label that
    the real feed does not have: there the ball moves at a median 2.8 m/s in stoppages. Each frozen
    spell is replaced by a ball carried to the restart: from where it froze, eased over
    FROZEN_BLEND_S onto the player nearest the ball at the first frame after the spell, then with
    that player.

    The real tracker also loses the ball in the keeper's hands, for 5-30 s after a catch. With keeper_hold on, a DFL copy hides the ball from
    each catch (keeper_holds(): ball high at the keeper in his box, live) until it leaves him, plus a re-acquisition
    delay, so the models can learn that a long gap there is live.
        python nootropia/training/make_benchmark.py calibrate > outputs/noise_profile.json
        python nootropia/training/make_benchmark.py build --profile outputs/noise_profile.json \
            metrica:1 metrica:2 metrica:3 dfl:J03WMX dfl:J03WN1
    `calibrate` reads the gated data locally and prints only aggregate statistics.

Everything is seeded: the same command writes the same files.
"""

import argparse
import csv
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

FPS = 25
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "games.py").is_file())  # the upstream repository
# Outputs go under the submission folder's outputs/, which is gitignored: the benchmark is hundreds of MB of
# third-party data and must not be committed. Absolute, so the script runs from any directory.
OUTPUTS = HERE.parent / "outputs"

# Public ground truth reference (DATA.md): 7,500 graded frames, D 1,469, 17 possessions of mean 14.2 s.
PUBLIC_GT_REF = {"d_share": 1469 / 7500, "mean_possession_s": 14.2}

NOISE_DEFAULTS = {
    "ball_missing": 0.41,        # share of graded frames with no ball (DATA.md: 41%)
    "ball_gap_mean_s": 1.2,      # mean length of a ball dropout burst
    "ball_gap_alive_mean_s": 0.0,  # > 0: mean dropout length in live play (BAR-ATH re-finds the ball in ~5 s), ball_gap_mean_s in D
    "ball_crowd_weight": 0.6,    # extra dropout hazard per player within 3 m of the ball
    "ball_fast_weight": 1.0,     # extra dropout hazard when the ball moves faster than 12 m/s
    # Odds of the ball missing on a D frame over the odds on an alive frame. Measured on the public
    # window by `calibrate` (BAR-ATH: missing on 34% of D frames, 6% of alive ones). 1 = dropout
    # ignores the ball state, which makes "no ball" worthless as a dead-ball cue.
    "ball_dead_odds_ratio": 8.0,
    "ball_out_tracked": 0.025,   # share of graded frames with the ball tracked beyond a line
    "ball_jitter_m": 0.25,       # size of the drifting ball position error
    "ball_jitter_tau_s": 0.4,    # how fast it drifts (Gaussian time scale)
    "ball_smooth_frames": 2,     # Gaussian smoothing of the source ball, in frames
    "tracklet_mean_s": 25.0,     # mean length of a player tracklet before the tracker loses it
    "player_gap_mean_s": 1.5,    # mean time a lost player stays unseen
    "occlusion_weight": 3.0,     # extra break hazard when another player is within 1.5 m
    "occlusion_scale_m": 0.0,    # > 0: the extra hazard is weight * exp(-gap / scale) instead (gap to the nearest player)
    # Keeper holds (DFL only, keeper_holds()): the BAR-ATH tracker loses the ball in the keeper's hands and the
    # models call it dead. 1 = hide the ball from the catch until it leaves him, then for a re-acquisition
    # delay drawn per hold (exponential, this mean): BAR-ATH's keeper gaps last 5-30 s.
    "keeper_hold": 0,
    "keeper_reacquire_mean_s": 3.0,
    # [lo, hi] s, DFL only (stretch_holds()): slow each hold down to a length drawn uniformly from this range, so
    # players walk while the keeper waits, as on BAR-ATH (DFL keepers release in ~1-5 s). None = off
    "keeper_stretch_s": None,
    # Box stoppages (box_stops()): the counterpart of the holds. Without it every ball hidden in a box is a live
    # hold, and a model learns "ball lost in the box = live" (BAR-ATH's goal kicks then reopen). Share of the
    # stoppages that start with the ball in a penalty area or over a goal line whose ball is hidden, from a start
    # drawn within box_stop_lead_s of the stoppage (seen going out or not) to the restart plus a re-acquisition
    # delay (keeper_reacquire_mean_s). 0 = off
    "box_stop_hide": 0.0,
    "box_stop_lead_s": 1.0,
    "team_error": 0.02,          # share of tracklets assigned to the wrong team
    "player_jitter_m": 0.15,     # size of the drifting player position error
    "player_jitter_tau_s": 1.0,  # how fast it drifts
    "player_smooth_frames": 2,   # Gaussian smoothing of the source players, in frames
    "seed": 0,
}


# ============================================================================ clean match
class Match:
    """Clean source data in metres: ball (N,2), home (N,Ph,2), away (N,Pa,2), labels (N,), period (N,)."""

    def __init__(self, name, ball, home, away, labels, period, pitch, teams, source, label_source):
        self.name, self.ball, self.home, self.away = name, ball, home, away
        self.labels, self.period, self.pitch, self.teams = labels, period, pitch, teams
        self.source, self.label_source = source, label_source
        self.holds = None  # [(catch, release)] keeper holds, DFL only (keeper_holds())


def _from_kloppy(ds, name, source, label_source):
    df = ds.to_df(engine="pandas").sort_values(["period_id", "frame_id"]).reset_index(drop=True)
    L, W = ds.metadata.pitch_dimensions.pitch_length, ds.metadata.pitch_dimensions.pitch_width
    teams = ds.metadata.teams  # [home, away]

    def team_xy(team):
        cols = [p.player_id for p in team.players if f"{p.player_id}_x" in df]
        x = df[[f"{c}_x" for c in cols]].to_numpy(float) * L
        y = df[[f"{c}_y" for c in cols]].to_numpy(float) * W
        return np.stack([x, y], -1)

    ball = np.stack([df.ball_x.to_numpy(float) * L, df.ball_y.to_numpy(float) * W], -1)
    return df, Match(name, ball, team_xy(teams[0]), team_xy(teams[1]), None,
                     df.period_id.to_numpy(int), (float(L), float(W)), [str(t) for t in teams],
                     source, label_source)


def merge_flicker(labels, k):
    """Give a possession run shorter than k frames to the other team when that team holds the ball
    directly before AND after it (H H A H H -> H H H H H). Runs next to a dead ball are kept."""
    if k <= 0:
        return labels, 0
    lab = labels.copy()
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(lab)]))
    merged = 0
    for i in range(1, len(starts) - 1):
        cur, before, after = lab[starts[i]], lab[starts[i - 1]], lab[starts[i + 1]]
        if cur in "HA" and before == after and before in "HA" and before != cur \
                and ends[i] - starts[i] < k:
            lab[starts[i]:ends[i]] = before
            merged += 1
    return lab, merged


def load_dfl(spec, min_spell=0):
    from kloppy import sportec
    if spec.startswith("dfl-files:"):
        meta, raw = spec.split(":", 1)[1].split(",")
        ds = sportec.load_tracking(meta_data=meta, raw_data=raw, only_alive=False, coordinates="metrica")
        stem = Path(raw).stem  # figshare names end in "..._DFL-MAT-<match id>": keep just the id
        name = "DFL-" + (stem.rsplit("DFL-MAT-", 1)[1] if "DFL-MAT-" in stem else stem)
    else:
        mid = spec.split(":", 1)[1]
        try:
            ds = sportec.load_open_tracking_data(match_id=mid, only_alive=False, coordinates="metrica")
        except Exception as exc:  # figshare answers scripted downloads with an empty bot challenge
            from kloppy._providers.sportec import get_IDSSE_url
            raise RuntimeError(
                f"could not download DFL match {mid} from figshare ({type(exc).__name__}: {exc}). "
                f"figshare blocks non-browser downloads: open {get_IDSSE_url(mid, 'meta')} (meta) and "
                f"{get_IDSSE_url(mid, 'tracking')} (positions) in a browser, then pass the saved files "
                f"as dfl-files:<meta.xml>,<positions.xml>") from exc
        name = "DFL-" + mid
    df, m = _from_kloppy(ds, name, "DFL/IDSSE open data (Bassek et al. 2025, CC-BY 4.0)",
                         "tracking BallStatus/BallPossession flags")
    home_id = ds.metadata.teams[0].team_id
    alive = (df.ball_state.astype(str) == "alive").to_numpy()
    owner = df.ball_owning_team_id.astype(str).to_numpy()
    raw_labels = np.where(~alive, "D", np.where(owner == home_id, "H", "A")).astype("<U1")
    m.labels, merged = merge_flicker(raw_labels, min_spell)
    if min_spell:
        m.label_source += f" (possession runs < {min_spell} frames between the same team merged: {merged})"
    m.holds = keeper_holds(df, ds.metadata.teams, m.pitch, alive)
    return m


HOLD_HANDS_M = 1.5   # ball within this of the keeper ...
HOLD_HANDS_Z = 0.5   # ... and this high (in his hands, not at his feet): a catch
HOLD_MIN = 10        # frames in his hands: shorter is a punch or a deflection
HOLD_LEAVE_M = 2.0   # the hold ends when the ball is this far from him (thrown, kicked, or played on)


def keeper_holds(df, teams, pitch, alive):
    """[(catch, release)] frames: the ball in a keeper's hands in his penalty area in live play, until it
    leaves him. Keepers from the metadata (starting position); DFL's ball height tells hands from feet.
    On J03WR9: 18 holds of 0.6-16 s, 100% live and 93-98% the keeper's team on DFL's flags."""
    L, W = pitch
    bx, by, bz = df.ball_x.to_numpy(float) * L, df.ball_y.to_numpy(float) * W, df.ball_z.to_numpy(float)
    holds, n = [], len(df)
    for team in teams:
        for p in team.players:
            if "oal" not in str(p.starting_position) or f"{p.player_id}_x" not in df:
                continue
            kx, ky = df[f"{p.player_id}_x"].to_numpy(float) * L, df[f"{p.player_id}_y"].to_numpy(float) * W
            d = np.hypot(bx - kx, by - ky)
            in_box = (np.minimum(kx, L - kx) < 16.5) & (np.abs(ky - W / 2) < 20.16)
            with np.errstate(invalid="ignore"):
                hands = alive & (d < HOLD_HANDS_M) & (bz > HOLD_HANDS_Z) & in_box
            f = 0
            while f < n:
                if not hands[f]:
                    f += 1
                    continue
                s = f
                while f < n and hands[f]:
                    f += 1
                if f - s < HOLD_MIN:
                    continue
                with np.errstate(invalid="ignore"):
                    while f < n and alive[f] and d[f] < HOLD_LEAVE_M:
                        f += 1
                holds.append((s, f))
    return sorted(holds)


def stretch_holds(m, prof):
    """Slow every keeper hold of m down, in place, to a length drawn from prof["keeper_stretch_s"] (never shorter
    than it was): the hold's frames are resampled over the longer span, so everyone moves slower by the same factor
    and the labels (live, mostly the keeper's team) run on. Returns float[new n], the source frame of each new
    frame (fractional inside holds), or None when off. Its own random stream, like the holds in degrade()."""
    if not (prof.get("keeper_stretch_s") and m.holds):
        return None
    lo, hi = prof["keeper_stretch_s"]
    rng = np.random.default_rng([prof["seed"], 2])
    n = len(m.ball)
    pieces, holds, prev, size = [], [], 0, 0
    for s, e in m.holds:
        if s < prev:  # overlaps the previous hold (both keepers near the ball): keep the first
            continue
        length = max(int(rng.uniform(lo, hi) * FPS), e - s)
        pieces += [np.arange(prev, s, dtype=float), s + np.arange(length) * (e - s) / length]
        size += s - prev
        holds.append((size, size + length))
        size += length
        prev = e
    src = np.concatenate(pieces + [np.arange(prev, n, dtype=float)])
    a = np.floor(src).astype(int)
    b, w = np.minimum(a + 1, n - 1), src - a

    def warp(xy):
        ww = w.reshape((-1,) + (1,) * (xy.ndim - 1))
        return (1 - ww) * xy[a] + ww * xy[b]  # NaN at either end stays NaN
    m.ball, m.home, m.away = warp(m.ball), warp(m.home), warp(m.away)
    m.labels, m.period, m.holds = m.labels[a], m.period[a], holds
    return src


BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16


def box_stops(m):
    """[(first, stop)] frames: stoppages (runs of D inside a period, not its start) whose ball, on the last live
    frame before them, is in a penalty area or beyond a goal line: goal kicks, corners, fouls and saves in the box."""
    L, W = m.pitch
    dead = m.labels == "D"
    stops, n, f = [], len(dead), 1
    while f < n:
        if not (dead[f] and not dead[f - 1] and m.labels[f - 1] != "X" and m.period[f] == m.period[f - 1]):
            f += 1
            continue
        e = f
        while e < n and dead[e]:
            e += 1
        x, y = m.ball[f - 1]
        if np.isfinite(x) and (min(x, L - x) < BOX_DEPTH_M) and (abs(y - W / 2) < BOX_HALF_WIDTH_M or x < 0 or x > L):
            stops.append((f, e))
        f = e
    return stops


# ---------------------------------------------------------------------------- Metrica (events)
POSS_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT"}


def _metrica_events(game_dir, game):
    """Events as dicts: team H/A, type, subs (set), period, start, end (frame ids)."""
    out = []
    csv_path = game_dir / f"Sample_Game_{game}_RawEventsData.csv"
    if csv_path.exists():
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                out.append({"team": {"Home": "H", "Away": "A"}.get(r["Team"]), "type": r["Type"],
                            "subs": set(filter(None, r["Subtype"].split("-"))), "period": int(r["Period"]),
                            "start": int(r["Start Frame"]), "end": int(r["End Frame"])})
        return out
    data = json.loads((game_dir / f"Sample_Game_{game}_events.json").read_text())["data"]
    meta =(game_dir / f"Sample_Game_{game}_metadata.xml").read_text()
    local = meta.split('idLocalTeam="', 1)[1].split('"', 1)[0]
    for e in data:
        st = e.get("subtypes") or []
        st = [st] if isinstance(st, dict) else st
        out.append({"team": "H" if e["team"]["id"] == local else "A", "type": e["type"]["name"],
                    "subs": {s["name"] for s in st}, "period": e["period"],
                    "start": e["start"]["frame"], "end": e["end"]["frame"] or e["start"]["frame"]})
    return out


def metrica_labels(events, frame_ids, period):
    """Per-frame H/A/D from Metrica events, following the challenge's possession rules.

    - Graded from each period's kick-off to its last event; X elsewhere.
    - Dead ball (D) from a stoppage trigger (ball out, shot out or goal, foul without advantage,
      offside, hand ball) until the next restart (any set piece, or a goal-kick pass). A trigger
      with no restart within 90 s is ignored (advantage played).
    - Otherwise the team of the latest on-ball event (pass, carry, shot, recovery, set piece,
      ball lost, ball out). Duels alone never change possession, and neither does the other team's
      "ball lost" / "ball out" (a team cannot gain the ball by losing it). A change is placed at a recovery
      or set piece when there is one, else when the previous team's last action ended (the ball
      reached the new team), never later than the new team's first event.
    """
    first_id = int(frame_ids[0])
    n = len(frame_ids)
    labels = np.full(n, "X", dtype="<U1")
    for p in sorted(set(period.tolist())):
        ev = sorted((e for e in events if e["period"] == p), key=lambda e: e["start"])
        if not ev:
            continue
        ko = next((e for e in ev if e["type"] == "SET PIECE" and "KICK OFF" in e["subs"]), ev[0])
        g0, g1 = ko["start"], max(max(e["start"], e["end"]) for e in ev)
        changes, prev = [], None
        for e in ev:
            if e["type"] not in POSS_TYPES or not e["team"]:
                continue
            if prev is not None and e["team"] != prev["team"] and e["type"] in ("BALL LOST", "BALL OUT"):
                continue  # losing the ball or putting it out never GAINS possession (often logged late)
            if prev is None:
                changes.append((e["start"], e["team"]))
            elif e["team"] != prev["team"]:
                pend = max(prev["start"], prev["end"])
                at = e["start"] if e["type"] in ("RECOVERY", "SET PIECE") or pend <= prev["start"] \
                    else min(e["start"], pend)
                changes.append((at, e["team"]))
            prev = e
        restarts = sorted(e["start"] for e in ev if e["type"] == "SET PIECE"
                          or (e["type"] == "PASS" and "GOAL KICK" in e["subs"]))
        dead = []
        for e in ev:
            t, s = e["type"], e["subs"]
            if t == "BALL OUT" or (t == "SHOT" and ({"OUT", "GOAL"} & s)) \
                    or (t == "BALL LOST" and ({"OFFSIDE", "HAND BALL"} & s)):
                trig = max(e["start"], e["end"])
            elif t == "FAULT RECEIVED" or (t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s):
                trig = e["start"]
            else:
                continue
            nxt = next((r for r in restarts if r > trig), None)
            if nxt is not None and nxt - trig <= 90 * FPS:
                dead.append((trig, nxt - 1))
        seg = np.full(g1 - g0 + 1, "D", dtype="<U1")
        for (a, team), (b, _) in zip(changes, changes[1:] + [(g1 + 1, None)]):
            lo, hi = max(a, g0), min(b, g1 + 1)
            if lo < hi:
                seg[lo - g0:hi - g0] = team
        for a, b in dead:
            seg[max(a, g0) - g0:min(b, g1) - g0 + 1] = "D"
        lo, hi = g0 - first_id, g1 - first_id
        labels[max(lo, 0):min(hi, n - 1) + 1] = seg[max(-lo, 0):len(seg) - max(hi - (n - 1), 0)]
    return labels


METRICA_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
METRICA_FILES = {
    "1": ["RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv", "RawEventsData.csv"],
    "2": ["RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv", "RawEventsData.csv"],
    "3": ["metadata.xml", "tracking.txt", "events.json"],
}


def fetch_metrica(game, metrica_dir):
    """Download the files one sample game needs (no git clone required). Skips files already there."""
    import urllib.request
    gd = Path(metrica_dir) / "data" / f"Sample_Game_{game}"
    gd.mkdir(parents=True, exist_ok=True)
    for suffix in METRICA_FILES[game]:
        name = f"Sample_Game_{game}_{suffix}"
        if not (gd / name).exists():
            print(f"  downloading {name} ...", flush=True)
            tmp = gd / (name + ".part")
            urllib.request.urlretrieve(f"{METRICA_RAW}/Sample_Game_{game}/{name}", tmp)
            tmp.replace(gd / name)
    return gd


def load_metrica(spec, metrica_dir):
    from kloppy import metrica
    game = spec.split(":", 1)[1]
    gd = fetch_metrica(game, metrica_dir)
    if (gd / f"Sample_Game_{game}_tracking.txt").exists():
        ds = metrica.load_tracking_epts(meta_data=str(gd / f"Sample_Game_{game}_metadata.xml"),
                                        raw_data=str(gd / f"Sample_Game_{game}_tracking.txt"),
                                        coordinates="metrica")
    else:
        ds = metrica.load_tracking_csv(home_data=str(gd / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv"),
                                       away_data=str(gd / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv"),
                                       coordinates="metrica")
    df, m = _from_kloppy(ds, f"METRICA-{game}", "Metrica Sports sample data", "derived from events")
    m.labels = metrica_labels(_metrica_events(gd, game), df.frame_id.to_numpy(int), m.period)
    return m


# ============================================================================ noise
def _speed(xy, k=2):
    v = np.full(xy.shape[:-1], np.nan)
    v[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) / (2 * k / FPS)
    return v


def _nan_smooth(x, sigma):
    """Gaussian smoothing over time (axis 0) that skips NaN: a normalised convolution, so a gap
    neither spreads nor drags its neighbours towards zero."""
    if sigma <= 0:
        return x
    ok = np.isfinite(x)
    num = gaussian_filter1d(np.where(ok, x, 0.0), sigma, axis=0, mode="constant")
    den = gaussian_filter1d(ok.astype(float), sigma, axis=0, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, num / den, np.nan)


def _drift(rng, shape, sd, tau_s):
    """Smooth random position error: white noise filtered over tau_s, rescaled to std sd."""
    tau = max(tau_s * FPS, 1e-9)
    return gaussian_filter1d(rng.normal(0, 1, shape), tau, axis=0) * sd * np.sqrt(2 * np.sqrt(np.pi) * tau)


def _wobble(xy):
    """Median second difference of positions (m) over finite steps: frame-to-frame roughness."""
    step = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=-1).ravel()
    step = step[np.isfinite(step) & (step < 2)]
    return round(float(np.median(step)), 4) if len(step) else None


def _outside(ball, pitch):
    """bool[n]: the ball is beyond a pitch line (False where it is missing)."""
    x, y = np.nan_to_num(ball[:, 0], nan=pitch[0] / 2), np.nan_to_num(ball[:, 1], nan=pitch[1] / 2)
    return (x < 0) | (x > pitch[0]) | (y < 0) | (y > pitch[1])


def _out_of_play_loss(outside, graded, target):
    """(bool[n] frames to drop, keep frames): every spell outside the lines keeps its first `keep`
    frames, the smallest keep for which tracked-outside frames reach `target` of graded frames."""
    edge = np.diff(np.concatenate(([0], outside.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    into = np.zeros(len(outside), int)  # frames since the spell started, for frames outside
    for s, e in zip(starts, ends):
        into[s:e] = np.arange(e - s)
    n_graded = max(graded.sum(), 1)
    longest = int((ends - starts).max()) if len(starts) else 0
    keep = 0  # a source that rarely has the ball outside (Metrica) never reaches the target: keep it all
    while keep < longest and (outside & graded & (into < keep)).sum() / n_graded < target:
        keep += 1
    return outside & (into >= keep), keep


FROZEN_MIN = 5         # frames of identical ball coordinates that make a frozen spell
FROZEN_BLEND_S = 2.0   # seconds to carry the ball from its frozen spot to the restarting player


def _unfreeze(ball, players):
    """(ball with every frozen spell replaced by a ball carried to the restarting player, frames
    replaced). Players are the source players, metres, NaN where absent."""
    ball = ball.copy()
    still = np.zeros(len(ball), bool)
    still[1:] = np.all(ball[1:] == ball[:-1], axis=1) & np.isfinite(ball[1:, 0])
    edge = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edge == 1) - 1, np.flatnonzero(edge == -1)  # spell = [start, end)
    replaced = 0
    for s, e in zip(starts, ends):
        if e - s < FROZEN_MIN or e >= len(ball) or not np.isfinite(ball[e, 0]):
            continue
        d = np.linalg.norm(players[e] - ball[e], axis=-1)
        if not np.isfinite(d).any():
            continue
        carrier = players[s:e, int(np.nanargmin(d))]
        carrier = pd_ffill_bfill(carrier)
        if not np.isfinite(carrier).all():
            continue
        w = np.clip(np.arange(e - s) / (FROZEN_BLEND_S * FPS), 0, 1)[:, None]
        ball[s:e] = (1 - w) * ball[s] + w * carrier
        replaced += e - s
    return ball, replaced


def pd_ffill_bfill(xy):
    """Fill NaN rows of an (n, 2) array forward, then backward."""
    out = xy.copy()
    for col in range(out.shape[1]):
        v = out[:, col]
        ok = np.isfinite(v)
        if not ok.any():
            continue
        idx = np.where(ok, np.arange(len(v)), 0)
        np.maximum.accumulate(idx, out=idx)
        v = v[idx]
        first = np.flatnonzero(ok)[0]
        v[:first] = out[first, col]
        out[:, col] = v
    return out


def _markov_dropout(hazard, p_rec, u):
    """Visible/lost chain: visible->lost w.p. hazard[t], lost->visible w.p. p_rec (scalar or per frame)."""
    lost = np.zeros(len(hazard), bool)
    p_rec = np.broadcast_to(p_rec, len(hazard))
    state = False
    for t in range(len(hazard)):
        state = (u[t] >= p_rec[t]) if state else (u[t] < hazard[t])
        lost[t] = state
    return lost


def _odds_ratio(lost, dead, graded):
    """Odds of a missing ball on graded D frames over the odds on graded alive frames."""
    def odds(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return p / (1 - p)
    return odds(lost[graded & dead].mean()) / odds(lost[graded & ~dead].mean())


def degrade(m, prof):
    rng = np.random.default_rng(prof["seed"])
    n = len(m.ball)
    graded = m.labels != "X"
    unfrozen, n_unfrozen = _unfreeze(m.ball, np.concatenate([m.home, m.away], axis=1))
    players = _nan_smooth(np.concatenate([m.home, m.away], axis=1), prof["player_smooth_frames"])
    source_ball = _nan_smooth(unfrozen, prof["ball_smooth_frames"])
    team = np.array([0] * m.home.shape[1] + [1] * m.away.shape[1])

    # --- ball: bursty dropout, likelier in crowds and on fast balls
    dist = np.linalg.norm(players - source_ball[:, None, :], axis=-1)
    crowd = np.nansum(dist < 3.0, axis=1)
    fast = np.nan_to_num(_speed(source_ball)) > 12.0
    mult = 1 + prof["ball_crowd_weight"] * crowd + prof["ball_fast_weight"] * fast
    p_rec = 1 / (prof["ball_gap_mean_s"] * FPS)
    u = rng.random(n)
    out_lost, out_keep = _out_of_play_loss(_outside(source_ball, m.pitch), graded,
                                           prof["ball_out_tracked"])
    already = np.isnan(m.ball[:, 0]) | out_lost
    held = np.zeros(n, bool)
    if prof.get("keeper_hold") and m.holds:
        # own stream: switching holds on leaves every other draw of the build as it was
        delays = np.random.default_rng([prof["seed"], 1]).exponential(
            prof["keeper_reacquire_mean_s"] * FPS, len(m.holds))
        for (s, e), extra in zip(m.holds, delays):
            held[s:min(e + int(extra), n)] = True
        already |= held
    hidden_stop, stops = np.zeros(n, bool), []
    if prof.get("box_stop_hide"):
        brng = np.random.default_rng([prof["seed"], 3])  # own stream, like the holds
        for s, e in box_stops(m):
            if brng.random() >= prof["box_stop_hide"]:
                continue
            a = max(s + int(brng.uniform(-1, 1) * prof["box_stop_lead_s"] * FPS), 0)
            z = min(e + int(brng.exponential(prof["keeper_reacquire_mean_s"] * FPS)), n)
            hidden_stop[a:z] = True
            stops.append((a, z))
        already |= hidden_stop
    dead = m.labels == "D"
    if prof.get("ball_gap_alive_mean_s"):  # BAR-ATH re-finds the ball faster in play (~5 s) than in stoppages
        p_rec = np.where(dead, p_rec, 1 / (prof["ball_gap_alive_mean_s"] * FPS))

    def fit_base(ratio):
        """Dropout with D-frame hazard `ratio` times the alive one, base hazard bisected so that
        ball_missing of graded frames have no ball."""
        lo, hi = 0.0, (1.0 if prof.get("ball_gap_alive_mean_s") else 0.2)  # fast recovery in play needs more
        for _ in range(20):
            mid = (lo + hi) / 2
            hazard = np.minimum(mid * mult * np.where(dead, ratio, 1.0), 1.0)
            lost = _markov_dropout(hazard, p_rec, u) | already
            if lost[graded].mean() < prof["ball_missing"]:
                lo = mid
            else:
                hi = mid
        return lost

    # Outer bisection on the hazard ratio (log scale) to hit the target odds ratio. Bursts are long,
    # so a gap started in a stoppage runs on into play: the ratio needed is larger than the odds
    # ratio it buys, and when the source already lacks the ball in dead time (Metrica) it can be < 1.
    lo, hi = -np.log(1000.0), np.log(1000.0)
    for _ in range(12):
        mid = (lo + hi) / 2
        lost = fit_base(np.exp(mid))
        if _odds_ratio(lost, dead, graded) < prof["ball_dead_odds_ratio"]:
            lo = mid
        else:
            hi = mid
    hazard_ratio = float(np.exp(mid))
    ball = source_ball + _drift(rng, source_ball.shape, prof["ball_jitter_m"], prof["ball_jitter_tau_s"])
    ball[lost] = np.nan

    # --- players: cut into tracklets, unseen gaps, wrong-team tracklets, jitter
    near = np.full(players.shape[:2], np.inf)
    for j in range(players.shape[1]):  # distance to the nearest other player (occlusion proxy)
        d = np.linalg.norm(players - players[:, j:j + 1], axis=-1)
        d[:, j] = np.inf
        near[:, j] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    base = 1 / (prof["tracklet_mean_s"] * FPS)
    if prof.get("occlusion_scale_m"):
        # smooth in the gap to the nearest player, as measured on BAR-ATH (outputs/domain/breaks_by_nn.txt):
        # isolated players are almost never lost, overlapping ones often
        crowd = np.exp(-np.where(np.isfinite(near), near, 1e3) / prof["occlusion_scale_m"])
    else:
        crowd = near < 1.5
    brk = rng.random(players.shape[:2]) < base * (1 + prof["occlusion_weight"] * crowd)
    gap_p = 1 / (prof["player_gap_mean_s"] * FPS)
    tracklets, n_wrong = [], 0  # (team, start, end, xy)
    for j in range(players.shape[1]):
        present = ~np.isnan(players[:, j, 0])
        t = 0
        while t < n:
            if not present[t]:
                t += 1
                continue
            s = t
            while t < n and present[t] and not (brk[t, j] and t > s):
                t += 1
            wrong = rng.random() < prof["team_error"]
            xy = players[s:t, j] + _drift(rng, (t - s, 2), prof["player_jitter_m"],
                                          prof["player_jitter_tau_s"])
            tracklets.append((int(1 - team[j] if wrong else team[j]), s, t, xy))
            n_wrong += wrong
            if t < n and present[t]:  # tracker lost the player: unseen for a while
                t += int(rng.geometric(gap_p))
    # pack tracklets into anonymous slots per team (a slot is a tracklet holder, not a person)
    slots_team, slot_rows = [], []
    for tm in (0, 1):
        heap = []
        for tr_team, s, e, xy in sorted((x for x in tracklets if x[0] == tm), key=lambda x: x[1]):
            if heap and heap[0][0] < s:
                _, k = heapq.heappop(heap)
            else:
                k = len(slots_team)
                slots_team.append(tm)
                slot_rows.append([])
            slot_rows[k].append((s, e, xy))
            heapq.heappush(heap, (e, k))
    pos = np.full((n, len(slots_team), 2), np.nan)
    for k, rows in enumerate(slot_rows):
        for s, e, xy in rows:
            pos[s:e, k] = xy
    stats = {
        "ball_missing_graded": round(float(np.isnan(ball[graded, 0]).mean()), 4),
        "ball_missing_dead": round(float(lost[graded & dead].mean()), 4),
        "ball_missing_alive": round(float(lost[graded & ~dead].mean()), 4),
        "ball_dead_odds_ratio": round(_odds_ratio(lost, dead, graded), 2),
        "ball_dead_hazard_ratio": round(hazard_ratio, 2),
        "mean_tracked_objects_graded": round(float((~np.isnan(pos[graded, :, 0])).sum(1).mean()), 2),
        "slots": len(slots_team),
        "tracklet_mean_s": round(float(np.mean([e - s for _, s, e, _ in tracklets]) / FPS), 1),
        "tracklets": len(tracklets),
        "wrong_team_tracklets": int(n_wrong),
        "ball_unfrozen_frames": int(n_unfrozen),
        "keeper_holds": len(m.holds) if held.any() else 0,
        "keeper_hold_frames": int(held.sum()),
        "keeper_hold_live": round(float((~dead[held]).mean()), 4) if held.any() else None,
        "keeper_hold_spans": [[int(s), int(e)] for s, e in m.holds] if held.any() else [],
        "box_stops": len(stops),
        "box_stop_frames": int(hidden_stop.sum()),
        "box_stop_dead": round(float(dead[hidden_stop].mean()), 4) if hidden_stop.any() else None,
        "box_stop_spans": [[int(a), int(z)] for a, z in stops],
        "ball_out_keep_s": round(out_keep / FPS, 2),
        "ball_out_tracked": round(float((_outside(ball, m.pitch) & graded).sum() / max(graded.sum(), 1)), 4),
        "player_wobble_m": _wobble(pos[graded]),
        "ball_wobble_m": _wobble(ball[graded][:, None]),
    }
    return ball, pos, np.array(slots_team), stats


# ============================================================================ writers
def label_stats(labels):
    g = labels[labels != "X"]
    runs, cur, durs = 0, None, []
    for lab in g:
        if lab in "HA":
            if lab != cur:
                runs += 1
                durs.append(0)
            durs[-1] += 1
        cur = lab
    return {"graded_frames": int(len(g)), "d_share": round(float((g == "D").mean()), 4),
            "home_share_alive": round(float((g == "H").sum() / max(1, (g != "D").sum())), 4),
            "possessions": runs, "mean_possession_s": round(float(np.mean(durs)) / FPS, 2) if durs else None,
            "public_gt_reference": PUBLIC_GT_REF}


def graded_blocks(labels, period):
    blocks = []
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero((period == p) & (labels != "X"))
        if len(idx):
            blocks.append([int(p), int(idx[0]), int(idx[-1])])
    return blocks


def write_meta(path, m, slots_team, period, n):
    L, W = m.pitch
    bounds = {p: (int(np.flatnonzero(period == p)[0]) + 1, int(np.flatnonzero(period == p)[-1]) + 1)
              for p in sorted(set(period.tolist()))}
    names = ["first_half", "second_half", "first_extra_half", "second_extra_half"]
    gparams = "".join(f"<ProviderParameter><Name>{names[p - 1]}_start</Name><Value>{a}</Value></ProviderParameter>"
                      f"<ProviderParameter><Name>{names[p - 1]}_end</Name><Value>{b}</Value></ProviderParameter>"
                      for p, (a, b) in bounds.items())
    teams = "".join(f'<Team id="T{i}"><Name>{nm}</Name><ProviderTeamsParameters><ProviderParameter>'
                    f'<Name>attack_direction_first_half</Name><Value>not_set</Value></ProviderParameter>'
                    f'</ProviderTeamsParameters></Team>' for i, nm in enumerate(m.teams[:2]))
    players = "".join(f'<Player id="S{k + 1}" teamId="T{t}"><Name>Track {k + 1}</Name>'
                      f'<ShirtNumber>{k + 1}</ShirtNumber></Player>' for k, t in enumerate(slots_team))
    chans = "".join(f'<PlayerChannel channelId="x" id="s{k + 1}_x" playerId="S{k + 1}"/>'
                    f'<PlayerChannel channelId="y" id="s{k + 1}_y" playerId="S{k + 1}"/>'
                    f'<PlayerChannel channelId="s" id="s{k + 1}_s" playerId="S{k + 1}"/>'
                    for k in range(len(slots_team)))
    groups = "".join(f'<SplitRegister separator=","><PlayerChannelRef playerChannelId="s{k + 1}_x"/>'
                     f'<PlayerChannelRef playerChannelId="s{k + 1}_y"/><PlayerChannelRef playerChannelId="s{k + 1}_s"/>'
                     f'</SplitRegister>' for k in range(len(slots_team)))
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<main><Metadata><GlobalConfig><FileName>{m.name}</FileName><TrackingType>Optical</TrackingType>
<ProviderName>Metrica Sports</ProviderName><FrameRate>{FPS}</FrameRate>
<ProviderGlobalParameters>{gparams}</ProviderGlobalParameters></GlobalConfig>
<Sessions><Session id="session1"><SessionType>Match</SessionType>
<FieldSize><Width>{round(L)}</Width><Height>{round(W)}</Height></FieldSize></Session></Sessions>
<Teams>{teams}</Teams><Players>{players}</Players>
<Devices><Device id="device1"><Name>Device 1</Name><Sensors>
<Sensor id="position"><Name>Position</Name><Channels><Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel></Channels></Sensor>
<Sensor id="speed"><Name>Speed</Name><Channels><Channel id="s"><Name>speed</Name><Unit>meters per second</Unit></Channel></Channels></Sensor>
</Sensors></Device></Devices><PlayerChannels>{chans}</PlayerChannels></Metadata>
<DataFormatSpecifications><DataFormatSpecification endFrame="{n}" separator=":" startFrame="1">
<StringRegister name="frameCount"/><SplitRegister separator=";">{groups}</SplitRegister>
<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="s"/></SplitRegister>
</DataFormatSpecification></DataFormatSpecifications></main>
"""
    Path(path).write_text(xml)


def _cols(vals, digits):
    return np.where(np.isnan(vals), "NaN", np.char.mod(f"%.{digits}f", np.nan_to_num(vals)))


def write_raw(path, ball, pos, pitch):
    L, W = pitch
    n, k = pos.shape[:2]
    bs = _speed(ball)
    ball_txt = np.char.add(np.char.add(np.char.add(_cols(ball[:, 0] / L, 5), ","),
                                       np.char.add(_cols(ball[:, 1] / W, 5), ",")), _cols(bs, 2))
    sp = _speed(pos)
    slot_txt = []
    for j in range(k):
        x, y, s = _cols(pos[:, j, 0] / L, 5), _cols(pos[:, j, 1] / W, 5), _cols(sp[:, j], 2)
        miss = np.isnan(pos[:, j, 0])
        s = np.where(miss, "NaN", s)
        slot_txt.append(np.char.add(np.char.add(np.char.add(x, ","), np.char.add(y, ",")), s))
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(f"{i + 1}:{';'.join(col[i] for col in slot_txt)}:{ball_txt[i]}\n")


def build(args):
    prof = dict(NOISE_DEFAULTS)
    if args.profile:
        prof.update({k: v for k, v in json.loads(Path(args.profile).read_text()).items() if k in prof})
    if args.seed is not None:
        prof["seed"] = args.seed
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"games": {}}
    failed = []
    for spec in args.sources:
        print(f"[{spec}] loading ...", flush=True)
        try:  # one source failing (usually a download) should not throw away the others
            m = load_metrica(spec, args.metrica_dir) if spec.startswith("metrica") \
                else load_dfl(spec, args.dfl_min_spell_frames)
        except Exception as exc:
            print(f"[{spec}] FAILED: {exc}", flush=True)
            failed.append(spec)
            continue
        source_frame = stretch_holds(m, prof)
        n = len(m.ball)
        print(f"[{spec}] {n} frames, {m.home.shape[1]}+{m.away.shape[1]} players; degrading ...", flush=True)
        ball, pos, slots_team, nstats = degrade(m, prof)
        gdir = out / m.name
        gdir.mkdir(exist_ok=True)
        stem = f"{m.name}_tactical"
        files = {"atd_meta": f"{m.name}/{stem}_atd.xml", "atd_raw": f"{m.name}/{stem}_atd.txt",
                 "gt": f"{m.name}/{m.name}_gt.csv"}
        if source_frame is not None:  # the source frame of each frame, for targets read from the source (ballnet)
            np.save(gdir / f"{m.name}_source_frame.npy", source_frame.astype(np.float32))
            files["source_frame"] = f"{m.name}/{m.name}_source_frame.npy"
            nstats["keeper_stretch_frames"] = int(n - (source_frame[-1] + 1))
        write_meta(gdir / f"{stem}_atd.xml", m, slots_team, m.period, n)
        write_raw(gdir / f"{stem}_atd.txt", ball, pos, m.pitch)
        with open(gdir / f"{m.name}_gt.csv", "w") as fh:
            fh.write("frame,label\n")
            fh.writelines(f"{i},{lab}\n" for i, lab in enumerate(m.labels))
        info = {"game": m.name, "source": m.source, "labels": m.label_source, "teams": m.teams,
                "n_frames": n, "fps": FPS, "pitch": list(m.pitch),
                "graded_blocks": graded_blocks(m.labels, m.period), "label_stats": label_stats(m.labels),
                "noise_profile": prof, "noise_applied": nstats, "files": files}
        (gdir / f"{m.name}_info.json").write_text(json.dumps(info, indent=2))
        manifest["games"][m.name] = info
        manifest_path.write_text(json.dumps(manifest, indent=2))
        ls = info["label_stats"]
        print(f"[{spec}] -> {gdir}  D {ls['d_share']:.1%}  {ls['possessions']} possessions of mean "
              f"{ls['mean_possession_s']} s  | ball missing {nstats['ball_missing_graded']:.1%} "
              f"(D {nstats['ball_missing_dead']:.0%} / alive {nstats['ball_missing_alive']:.0%}, odds ratio "
              f"{nstats['ball_dead_odds_ratio']} for target {prof['ball_dead_odds_ratio']}), "
              f"{nstats['mean_tracked_objects_graded']} objects/frame, {nstats['slots']} slots", flush=True)
        if ls["mean_possession_s"] is not None and ls["mean_possession_s"] < 4 \
                and ls["graded_frames"] > 10 * 60 * FPS:  # only meaningful on a real match
            print(f"[{spec}] WARNING: possessions average {ls['mean_possession_s']} s (expected ~10-15 s). "
                  f"The source labels flicker; rebuild with --dfl-min-spell-frames 25.", flush=True)
    if failed:
        sys.exit(f"failed sources: {' '.join(failed)}")


# ============================================================================ calibrate
def calibrate(args):
    """Aggregate noise statistics of the real challenge ATD (graded frames only). Prints JSON."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "evaluation"))
    from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
    import possession_metrics as pm  # noqa: E402  (upstream evaluation/possession_metrics.py)
    raw = require(["atd_raw"], GAME)[0]
    missing_at = np.zeros(GAMES[GAME]["n_video_frames"], bool)
    graded = set()
    for _, a, b in half_video_blocks(GAME):
        graded.update(range(a, b + 1))
    ball_missing, objects, gaps, cur_gap = [], [], [], 0
    player_xy, ball_xy = [], []  # graded positions in metres, for the wobble
    length, width = GAMES[GAME]["pitch"]
    run_len, runs = None, []
    with open(raw) as fh:
        for line in fh:
            fid, players, ball = line.rstrip("\n").split(":")
            vf = int(fid) - 1
            groups = players.split(";")
            vis = np.array([g[0] != "N" for g in groups])
            if run_len is None:
                run_len = np.zeros(len(groups), int)
            ended = (~vis) & (run_len > 0)
            runs.extend(run_len[ended].tolist())
            run_len = np.where(vis, run_len + 1, 0)
            if vf not in graded:
                continue
            miss = ball[0] == "N"
            player_xy.append([[float(v) for v in g.split(",")[:2]] if g[0] != "N" else [np.nan, np.nan]
                              for g in groups])
            ball_xy.append([float(v) for v in ball.split(",")[:2]] if not miss else [np.nan, np.nan])
            missing_at[vf] = miss
            ball_missing.append(miss)
            objects.append(int(vis.sum()))
            if miss:
                cur_gap += 1
            elif cur_gap:
                gaps.append(cur_gap)
                cur_gap = 0
    # public window: how much likelier the ball is to be missing in a stoppage than in play
    gt = pm.label_codes(pm.read_label_csv(ROOT / "evaluation" / "gt" / f"{GAME}_possession_gt_public.csv"))
    window = np.zeros(len(gt), bool)
    for a, b in pm.valid_runs(gt):
        window[a:b] = True
    pub_dead = gt == pm.D
    obj = float(np.mean(objects))
    tracklet_s = float(np.mean(runs)) / FPS
    vis_share = min(obj / 22.0, 0.99)
    graded_ball = np.array(ball_xy) * (length, width)
    prof = {
        "ball_missing": round(float(np.mean(ball_missing)), 4),
        "ball_out_tracked": round(float(_outside(graded_ball, (length, width)).mean()), 4),
        "ball_gap_mean_s": round(float(np.mean(gaps)) / FPS, 2),
        "tracklet_mean_s": round(tracklet_s, 1),
        "player_gap_mean_s": round(tracklet_s * (1 - vis_share) / vis_share, 2),
        "ball_dead_odds_ratio": round(_odds_ratio(missing_at, pub_dead, window), 2),
        "_measured": {"public_ball_missing_dead": round(float(missing_at[window & pub_dead].mean()), 4),
                      "public_ball_missing_alive": round(float(missing_at[window & ~pub_dead].mean()), 4),
                      "mean_tracked_objects": round(obj, 2), "slots": int(len(run_len)),
                      "ball_gap_median_s": round(float(np.median(gaps)) / FPS, 2),
                      "ball_gap_p90_s": round(float(np.percentile(gaps, 90)) / FPS, 2),
                      "tracklet_median_s": round(float(np.median(runs)) / FPS, 1),
                      "player_wobble_m": _wobble(np.array(player_xy) * (length, width)),
                      "ball_wobble_m": _wobble(np.array(ball_xy)[:, None] * (length, width))},
    }
    print(json.dumps(prof, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="convert source matches into benchmark games")
    b.add_argument("sources", nargs="+", help="metrica:1 metrica:2 metrica:3 dfl:J03WMX dfl-files:meta,raw ...")
    b.add_argument("--metrica-dir", default=str(OUTPUTS / "ext" / "sample-data"),
                   help="where the Metrica sample files live (downloaded there if missing)")
    b.add_argument("--profile", help="noise profile JSON (from calibrate)")
    b.add_argument("--out", default=str(OUTPUTS / "benchmark"))
    b.add_argument("--seed", type=int, help="noise seed (default: the profile's, else 0); the training copies "
                   "of possession/evidence.NOISE_SEEDS are built with --seed N --out outputs/benchmark_sN")
    b.add_argument("--dfl-min-spell-frames", type=int, default=0,
                   help="merge DFL possession flickers shorter than this many frames (0 = raw labels)")
    sub.add_parser("calibrate", help="measure the real ATD's noise (aggregate stats only)")
    args = ap.parse_args()
    build(args) if args.cmd == "build" else calibrate(args)


if __name__ == "__main__":
    main()
