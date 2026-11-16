"""The referee's whistle, from the tactical video's own audio: a BAR-ATH-only stoppage cue.

The tactical MP4 carries frame-aligned audio. On the kick-offs the referee's whistle is two steady tones
at ~3.9 and ~4.3 kHz, 0.3-1 s long; crowd singing in the same band glides in pitch. Per video frame (one
STFT column per 40 ms, 128 ms window, 16 kHz mono) the score is the weaker of the two narrow-band peaks
(LOW_TONE, HIGH_TONE) over the median of the whole band (BAND), in dB. A whistle is a run of that score
(smoothed over 3 frames) above THRESH_DB for at least MIN_LEN frames with both tone frequencies steady;
detections closer than MERGE frames merge. ~0.4 detections per graded minute.

It is used in two ways, in decode.py:
- pull_back: a decoded stoppage that starts within decode.WHISTLE_PULL after a whistle starts at the whistle
  instead (the referee stopped play there; the tracking only shows it once the players slow down). Checked
  without labels: on the frames this turns dead, the broadcast is off the live picture 55% of the time
  (decoded dead 67%, live 10%).
- short_stops: a whistle in live play with no decoded stoppage after it (a short stoppage the decoder
  missed) starts one, ending at short_stop_end(): the earliest of the players speeding up again
  (tagging.find_restart) and the tracked ball moving again (> BALL_GO for BALL_HOLD frames). Only if the
  players slowed to stoppage speed (< STOP_SPEED) and it lasts MIN_STOP. The end rule was chosen on DFL:
  from the true start of 216 on-pitch stoppages it is within 1 s of the true restart 39% of the time
  (speed rise alone 31%), > 2 s late 3% (5%); the speed rise alone lags (a short free kick is passed before
  the players speed up).

    frames = whistle.detect()    # int[] first frame of each whistle, cached
    cues = whistle.cues()        # int[N, 2]: whistle frame, end of its short stoppage (-1: none)
"""

import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd

from .loaders import CACHE_DIR, FPS, _cached  # first: puts the upstream modules on sys.path
from games import GAME, GAMES, video_path  # noqa: E402

# Params
SR, HOP, WIN = 16000, 640, 2048     # Hz; one STFT column per 40 ms video frame, 128 ms window
BAND = (2500, 4600)                  # Hz: the band the score's median is taken over
LOW_TONE = (3750, 4050)              # Hz: the whistle's lower tone (~3.9 kHz on the kick-offs)
HIGH_TONE = (4150, 4450)             # Hz: its upper tone (~4.3 kHz)
THRESH_DB = 14.0                     # weaker tone peak over the band median: ~0-5 dB on crowd, 20-30 on whistles
MIN_LEN = 5                          # frames (0.2 s)
MAX_HZ_STD = 40.0                    # Hz: both tones steady over the run (crowd singing glides)
MERGE = 25                           # frames: closer detections are one whistle
MIN_STOP = 2 * FPS                   # frames: a short stoppage lasts at least this
SEARCH = 30 * FPS                    # frames after the whistle to look for the restart
STOP_SPEED = 1.6                     # m/s: players' 1 s mean speed must fall below this (tagging.ALIVE_SPEED:
                                     # ~2.2 in play, ~1.2 in stoppages), else play went on
BALL_GO, BALL_HOLD = 5.0, 5          # m/s, frames: the tracked ball moving again ends the stoppage


def scores(game=GAME):
    """(dual_db, low_hz, high_hz) float32[n_video_frames]: the score and the two peak frequencies per
    frame, streamed from the video's audio with ffmpeg."""
    import imageio_ffmpeg
    n_frames = GAMES[game]["n_video_frames"]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", video_path(game),
           "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    lo = (freqs >= LOW_TONE[0]) & (freqs <= LOW_TONE[1])
    hi = (freqs >= HIGH_TONE[0]) & (freqs <= HIGH_TONE[1])
    window = np.hanning(WIN).astype(np.float32)
    out = np.full((3, n_frames + 1000), np.nan, np.float32)
    buf = np.zeros(WIN // 2, np.float32)  # left padding: column f is centred on frame f
    f = 0
    while True:
        raw = proc.stdout.read(SR * 120 * 2)
        if not raw:
            break
        buf = np.concatenate([buf, np.frombuffer(raw, np.int16).astype(np.float32) / 32768])
        n = (len(buf) - WIN) // HOP + 1
        if n <= 0:
            continue
        idx = np.arange(n)[:, None] * HOP + np.arange(WIN)[None, :]
        spec = np.abs(np.fft.rfft(buf[idx] * window, axis=1)) ** 2
        med = np.median(spec[:, band], 1) + 1e-12
        out[0, f:f + n] = 10 * np.log10(np.minimum(spec[:, lo].max(1), spec[:, hi].max(1)) / med + 1e-12)
        out[1, f:f + n] = freqs[lo][spec[:, lo].argmax(1)]
        out[2, f:f + n] = freqs[hi][spec[:, hi].argmax(1)]
        f += n
        buf = buf[n * HOP:]
    proc.wait()
    return tuple(out[:, :n_frames])


def detect(game=GAME):
    """int[]: the first frame of each whistle. Cached under the detector settings."""
    def build():
        dual, low, high = scores(game)
        smooth = pd.Series(dual).rolling(3, center=True, min_periods=1).mean().to_numpy()
        on = np.nan_to_num(smooth, nan=-99.0) > THRESH_DB
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        runs = []
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < MIN_LEN or np.std(low[a:b]) > MAX_HZ_STD or np.std(high[a:b]) > MAX_HZ_STD:
                continue
            if runs and a - runs[-1][1] < MERGE:
                runs[-1] = (runs[-1][0], b)
            else:
                runs.append((a, b))
        return np.array([a for a, _ in runs], dtype=int)
    settings = json.dumps([game, SR, HOP, WIN, BAND, LOW_TONE, HIGH_TONE, THRESH_DB, MIN_LEN, MAX_HZ_STD, MERGE])
    key = hashlib.sha1(settings.encode()).hexdigest()[:8]
    return _cached(os.path.join(CACHE_DIR, "whistles_{}.pkl".format(key)), build)


def short_stop_end(w, speed_1s, ball_speed):
    """End (first live frame) of the short stoppage a whistle at w would start, or -1 (module doc).
    speed_1s: players' 1 s mean speed; ball_speed: tracked ball speed, m/s (features.py)."""
    from . import tagging
    a = tagging.find_restart(speed_1s, w + MIN_STOP, search=SEARCH - MIN_STOP)
    moving = pd.Series(np.nan_to_num(ball_speed, nan=0.0) > BALL_GO).rolling(BALL_HOLD).sum().to_numpy() == BALL_HOLD
    i = np.flatnonzero(moving[w + MIN_STOP:w + SEARCH])
    b = w + MIN_STOP + i[0] - (BALL_HOLD - 1) if len(i) else None
    end = min([r for r in (a, b) if r is not None], default=None)
    if end is None:
        return -1
    seg = speed_1s[w:end]
    return int(end) if np.isfinite(seg).any() and np.nanmin(seg) < STOP_SPEED and end - w >= MIN_STOP else -1


def cues(game=GAME):
    """int[N, 2]: each whistle (detect()) and the end of the short stoppage it would start (-1: none)."""
    from .features import load_features
    feats = load_features()
    speed_1s, ball_speed = feats["speed_1s"].to_numpy(float), feats["ball_speed"].to_numpy(float)
    frames = detect(game)
    return np.array([[w, short_stop_end(w, speed_1s, ball_speed)] for w in frames], dtype=int).reshape(-1, 2)
