"""Decoder: from per-frame evidence to H/A/D labels.

Viterbi over a cost per frame and state (negative log-probability of the evidence) plus a cost per
change of state, so a switch only happens when the evidence for the new state outweighs it. Two
passes: Live/Dead over each graded half, then H/A inside each Live run. Team evidence never touches
the alive/dead decision, and ownership is decided afresh after a stoppage.

Cost unit: one frame of fully confident evidence at weight 1 costs -log(CLIP) (~3 for 0.05).

Edge hints (dead.edges()): frames where a stoppage likely starts (start hint) or play likely
restarts (end hint), from sharp cues. In the two-pass decoder a hint lowers the Live -> Dead or
Dead -> Live switch cost there by up to HINT_DISCOUNT, so when the evidence supports a stoppage its
edges snap to the cue instead of wherever p_dead crosses over. The restarting team is matched to
the decoder's own restarts (within RESTART_MATCH), not to wherever p_dead crosses 0.5.
"""

from dataclasses import dataclass

import numpy as np

from .loaders import FPS  # first: puts the upstream modules on sys.path
import baseline  # noqa: E402  (upstream starter/baseline.py)
from games import GAME, half_video_blocks  # noqa: E402

# Params
CLIP = 0.05             # probabilities are clipped to [CLIP, 1 - CLIP]: one frame never forces a switch
W_CTRL = 1.0            # weight of p_home per frame
W_TAG = 0.3             # weight of tag_team per frame
W_DEAD = 2.0            # weight of p_dead per frame, tuned on the held-out DFL benchmark: the
                        # middle of the plateau with LAMBDA_DEAD (spell F1 within 0.01 of the best)
W_RESTART = 2.0         # per frame against the team that is not restarting
RESTART_FRAMES = 25     # how long after a restart that cost applies
LAMBDA_TEAM = 2.0       # cost of an H <-> A switch
                        # Model-specific: retune after any retrain.
LAMBDA_DEAD = 150.0     # cost of entering or leaving D
HINT_DISCOUNT = 0.8     # share of LAMBDA_DEAD a full-strength edge hint takes off the switch
WHISTLE_PULL = 20 * FPS  # frames: a decoded stoppage starting this soon after a referee whistle (whistle.py)
                        # starts at the whistle.
WHISTLE_GUARD = 5 * FPS  # frames: a whistle this soon after a decoded stoppage is the restart's, not a new stoppage
RESTART_MATCH = 50      # frames: a restart_team mark this close to a decoded restart applies to it
KICKOFF_2ND = 113742    # BAR-ATH second-half kick-off (FAQ.md): its graded block starts at 110808, and
                        # the 2934 frames before the kick-off are all D. The first half is graded from
                        # its kick-off. (games.py's kickoff_2nd anchor reads 113741, one frame early.)


@dataclass(frozen=True)
class Params:
    """Decoder settings for one call. None means the module constant of the same name in capitals,
    read when decode() runs."""
    lambda_team: float | None = None
    w_tag: float | None = None
    w_dead: float | None = None
    lambda_dead: float | None = None
    hint_discount: float | None = None


def _resolve(params):
    params = params or Params()
    return Params(**{k: globals()[k.upper()] if v is None else v for k, v in vars(params).items()})


LIVE, DEAD = 0, 1
H, A, D = 0, 1, 2       # state indices; also index LABELS
LABELS = np.array(["H", "A", "D"])


def _nll(p, weight):
    """float[N, 2]: weighted cost of the evidence p for (state True, state False). 0 where NaN."""
    p = np.clip(p, CLIP, 1 - CLIP)
    cost = -weight * np.log(np.stack([p, 1 - p], axis=1))
    cost[np.isnan(p)] = 0.0
    return cost


def team_costs(p_home, restart_team, tag_team, w_tag=None):
    """float[N, 2]: cost of H and of A per frame, shifted so the cheaper team costs 0.

    The shift keeps team evidence neutral about D: a confident nearest player at a throw-in makes
    one team cheap, it does not make the ball alive.
    """
    cost = _nll(p_home, W_CTRL) + _nll(tag_team, W_TAG if w_tag is None else w_tag)
    for frame in np.flatnonzero(restart_team != -1):
        other = A if restart_team[frame] == baseline.HOME else H
        cost[frame:frame + RESTART_FRAMES, other] += W_RESTART
    return cost - cost.min(axis=1, keepdims=True)


def dead_costs(p_dead, w_dead=None):
    """float[N, 2]: cost of Live and of Dead per frame."""
    return _nll(p_dead, W_DEAD if w_dead is None else w_dead)[:, ::-1]


def viterbi(costs, switch, start=None):
    """int[N]: the state sequence of least total cost.

    costs: float[N, S] per-frame cost of each state; switch: float[S, S], switch[i, j] = cost of
    going from i to j, or float[N, S, S] for a cost that changes per frame (switch[t] applies to the
    step into frame t); start: float[S] cost of starting in each state (default 0).
    """
    n, s = costs.shape
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    ptr = np.empty((n, s), dtype=np.int8)
    v = costs[0] + (np.zeros(s) if start is None else start)
    for t in range(1, n):
        cand = v[:, None] + (switch[t] if switch.ndim == 3 else switch)   # cand[from, to]
        ptr[t] = cand.argmin(axis=0)
        v = cand[ptr[t], np.arange(s)] + costs[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = v.argmin()
    for t in range(n - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    return path


def _switch(cost):
    """2x2 switch matrix with `cost` off the diagonal."""
    return np.array([[0.0, cost], [cost, 0.0]])


def _runs_of(mask):
    """[(start, stop)] of the True runs of a bool array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _hinted_switch(start_hint, end_hint, lambda_dead, hint_discount):
    """float[n, 2, 2] Live/Dead switch costs, lowered where the edge hints are."""
    switch = np.zeros((len(start_hint), 2, 2))
    switch[:, LIVE, DEAD] = lambda_dead * (1 - hint_discount * start_hint)
    switch[:, DEAD, LIVE] = lambda_dead * (1 - hint_discount * end_hint)
    return switch


def team_pass(team, live_runs, restart_team=None, params=None):
    """int8[N] H/A state indices inside each live run [(start, stop)], D elsewhere: the second pass
    of decode_two_pass, callable on any live runs.

    restart_team: int8[N] marks (baseline.HOME / AWAY, -1 elsewhere); the mark nearest a run's
    start, within RESTART_MATCH, makes the other team pay W_RESTART for RESTART_FRAMES.
    """
    switch = _switch(_resolve(params).lambda_team)
    states = np.full(len(team), D, dtype=np.int8)
    marks = np.flatnonzero(restart_team != -1) if restart_team is not None else np.zeros(0, int)
    for a, b in live_runs:
        cost = team[a:b].copy()
        near = marks[np.abs(marks - a) <= RESTART_MATCH]
        if len(near):
            restarting = restart_team[near[np.argmin(np.abs(near - a))]]
            cost[:RESTART_FRAMES, A if restarting == baseline.HOME else H] += W_RESTART
        states[a:b] = viterbi(cost, switch)
    return states


def pull_back(live, whistles, window=WHISTLE_PULL):
    """bool[n] live with each stoppage that starts within `window` frames after a whistle (local frame
    indices, whistles inside a live stretch) started at the whistle instead."""
    live = live.copy()
    for w in whistles:
        if 0 <= w < len(live) and live[w]:
            stop = np.flatnonzero(~live[w:w + window])
            if len(stop):
                live[w:w + stop[0]] = False
    return live


def short_stops(live, whistles, ends, window=WHISTLE_PULL, guard=WHISTLE_GUARD):
    """bool[n] live with a short stoppage [whistle, end) at each whistle in live play with no stoppage
    starting within `window` after it (pull_back's case) or ending within `guard` before it (a restart
    whistle). whistles, ends: local frame indices (end < 0: the whistle starts none; whistle.cues())."""
    out = live.copy()
    for w, e in zip(whistles, ends):
        if e < 0 or not (0 <= w < len(live)) or not live[w]:
            continue
        if (~live[w:w + window]).any() or (~live[max(w - guard, 0):w]).any():
            continue
        out[w:min(e, len(live))] = False
    return out


def decode_two_pass(team, dead, runs, hints=None, restart_team=None, params=None, start_live=False, whistles=None):
    """int8[N] H/A/D state indices: Live/Dead first, then H/A inside each Live run (team_pass).

    hints: (start_hint, end_hint) float[N] in 0..1, or None for a constant switch cost.
    restart_team: see team_pass. params: Params (None: the module constants).
    start_live: each run starts at its kick-off, so being dead on its first frame costs the switch
    (BAR-ATH: bar_ath_runs()). Default False: the run may start before play, so it starts dead.
    whistles: referee whistles, int[N] frames or int[N, 2] (frame, short stoppage end; whistle.cues()):
    pull_back(), then short_stops(), after the Live/Dead pass.
    """
    params = _resolve(params)
    lam = params.lambda_dead
    start = np.array([0.0, lam]) if start_live else np.array([lam, 0.0])
    live_runs = []
    for first, stop in runs:
        switch = (_switch(lam) if hints is None
                  else _hinted_switch(hints[0][first:stop], hints[1][first:stop], lam, params.hint_discount))
        live = viterbi(dead[first:stop], switch, start) == LIVE
        if whistles is not None and len(whistles):
            cues = np.asarray(whistles, dtype=int)
            cues = cues if cues.ndim == 2 else np.column_stack([cues, np.full(len(cues), -1)])
            frames = cues[:, 0] - first
            pulled = pull_back(live, frames)
            live = short_stops(pulled, frames, np.where(cues[:, 1] >= 0, cues[:, 1] - first, -1))
        live_runs += [(first + a, first + b) for a, b in _runs_of(live)]
    return team_pass(team, live_runs, restart_team, params)


def bar_ath_runs():
    """[(first, stop)] BAR-ATH's graded halves, each from its kick-off (see KICKOFF_2ND)."""
    return [(max(first, KICKOFF_2ND) if half == 2 else first, last + 1)
            for half, first, last in half_video_blocks(GAME)]


def decode(p_home, p_dead, restart_team, tag_team, runs=None, hints=None, params=None, whistles=None,
           start_live=None):
    """str[N] H/A/D labels from the evidence arrays of lines 1, 2 and 4.

    runs: [(first, stop)] graded runs to decode, each independently (default: bar_ath_runs()).
    Frames outside them are D. hints: dead.edges(). params: Params (None: the module constants).
    whistles: referee whistle frames (whistle.cues()); None on the benchmark (no audio). start_live: each run starts at its kick-off
    (decode_two_pass); None: True when runs is None (BAR-ATH's halves), else False.
    """
    params = _resolve(params)
    if start_live is None:
        start_live = runs is None      # bar_ath_runs() start each half at its kick-off
    if runs is None:
        runs = bar_ath_runs()
    restart_team = np.asarray(restart_team)
    dead = dead_costs(np.asarray(p_dead, float), params.w_dead)
    # the restart team is applied at the decoded restarts (team_pass), not as a per-frame cost
    team = team_costs(np.asarray(p_home, float), np.full(len(restart_team), -1),
                      np.asarray(tag_team, float), params.w_tag)
    return LABELS[decode_two_pass(team, dead, runs, hints, restart_team, params, start_live, whistles)]
