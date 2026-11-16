"""Off-screen freeze for a benchmark copy

Used only for training (train_ballnet.py).
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from possession import dead, decode  # first: puts the upstream modules on sys.path
from possession.loaders import load_frame_df
from games import GAME, GAMES  # noqa: E402
import load_homography as lh  # noqa: E402  (upstream starter/load_homography.py)

# Params
W_PX, H_PX = 1920, 1080
LAG = 37            # frames: the camera follows the ball ~1.5 s late (correlation peaks at 25-50)
SMOOTH = 50         # frames: and smoothly
ACTIVE_SPEED = 1.5  # m/s: BAR-ATH frames used to fit the camera have players moving (1 s mean speed)
MARGIN_PX = 250     # px beyond the image before a player is frozen: gives the copies 4.5-6% of tracked
                    # players frozen, against BAR-ATH's 4.7% off-screen (the real tracker sees a little wider)
_BANK = []


def project(H, xn, yn):
    """Pixels (u, v) of normalised pitch points; H [N, 3, 3], xn / yn [N, S]."""
    d = H[:, 2, 0, None] * xn + H[:, 2, 1, None] * yn + H[:, 2, 2, None]
    u = (H[:, 0, 0, None] * xn + H[:, 0, 1, None] * yn + H[:, 0, 2, None]) / d
    v = (H[:, 1, 0, None] * xn + H[:, 1, 1, None] * yn + H[:, 1, 2, None]) / d
    return u, v


def _interp(a):
    idx = np.arange(len(a))
    ok = np.isfinite(a)
    return np.interp(idx, idx[ok], a[ok]) if ok.any() else a


def camera_target(bx_norm):
    """Where the camera points (pitch x, 0-1): the ball, filled through gaps, smoothed and lagged."""
    s = uniform_filter1d(_interp(bx_norm), SMOOTH)
    return np.r_[np.full(LAG, s[0]), s[:-LAG]]


def camera_bank():
    """BAR-ATH's active graded homographies sorted by view-centre x, and the fit centre ~ a + b * target.
    Cached in memory."""
    if not _BANK:
        frame_df, _, _ = load_frame_df()
        length = GAMES[GAME]["pitch"][0]
        bx = frame_df["ball_x"].to_numpy(float) * length
        H, _, _ = lh.load()
        live = np.zeros(len(frame_df), bool)
        for a, b in decode.bar_ath_runs():
            live[a + 1:b - 1] = True
        live &= dead.bar_ath_sequence()["speed_1s"].to_numpy(float) >= ACTIVE_SPEED
        c = np.linalg.inv(H) @ np.array([W_PX / 2, H_PX / 2, 1.0])
        cx = c[:, 0] / c[:, 2]
        ok = live & np.isfinite(cx)
        b, a = np.polyfit(camera_target(bx / length)[ok], cx[ok], 1)
        order = np.argsort(cx[ok])
        _BANK.append((H[ok][order], cx[ok][order], (a, b)))
    return _BANK[0]


def copy_camera(bx_norm):
    """H [N, 3, 3]: a BAR-ATH homography per frame of a copy, from its ball x (0-1, NaN = not seen)."""
    Hs, cxs, (a, b) = camera_bank()
    k = np.clip(np.searchsorted(cxs, a + b * camera_target(bx_norm)), 0, len(cxs) - 1)
    return Hs[k]


def freeze(x, y, ball_x, pitch, chunk=20000):
    """(x, y) of a copy with off-screen players held at their last on-screen position, and the share of
    tracked slot-frames held. x, y float[N, S] metres (NaN untracked); ball_x float[N] metres."""
    length, width = pitch
    H = copy_camera(ball_x / length)
    edge = np.full(x.shape, np.nan)
    for s in range(0, len(x), chunk):
        u, v = project(H[s:s + chunk], x[s:s + chunk] / length, y[s:s + chunk] / width)
        edge[s:s + chunk] = np.minimum(np.minimum(u, W_PX - u), np.minimum(v, H_PX - v))
    tracked = np.isfinite(x)
    on = tracked & (edge >= -MARGIN_PX)
    idx = np.arange(len(x))[:, None]
    last_on = np.maximum.accumulate(np.where(on, idx, -1), axis=0)
    # a new tracklet id at every start of tracking, so a hold never crosses a gap
    start = tracked & ~np.r_[np.zeros((1, x.shape[1]), bool), tracked[:-1]]
    tid = np.cumsum(start, axis=0)
    hold = tracked & ~on & (last_on >= 0)
    hold &= np.take_along_axis(tid, np.maximum(last_on, 0), axis=0) == tid
    src = np.where(hold, last_on, idx)
    return np.take_along_axis(x, src, axis=0), np.take_along_axis(y, src, axis=0), hold.sum() / max(tracked.sum(), 1)
