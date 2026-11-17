"""The broadcast recording's tracking placed on the tactical frames.

The broadcast is the live TV feed, so within a half it runs at a constant delay from the tactical
recording. The delay is found from the two trackings alone (no video pixels, no labels): the lag
that minimises the median distance between the tactical and the broadcast ball, first on 1 s means
over a wide range, then frame by frame around the best coarse lag.

Two things come from it:
- **ball gap-fill**: where the tactical ball is missing and the broadcast has a trustworthy pitch
  transform (valid, reprojection error below MAX_REPROJ), the broadcast ball is used;
- **cut-away flag**: where the broadcast has no pitch transform at the aligned frame, the director
  is showing a replay, a close-up or the crowd, which happens mostly while the ball is dead.
"""

import json
import os

import numpy as np

from . import sources

MAX_REPROJ = 5.0
MAX_JUMP_M = 3.0
COARSE = 25


def _parse_ball(raw_path):
    """(T, 3) normalised ball from a broadcast ATD (its player-slot count differs from the
    tactical file's, so only the last field, the ball, is read); frame ids checked."""
    import io
    import pandas as pd
    with open(raw_path) as handle:
        text = handle.read()
    frames, balls = [], []
    for line in text.splitlines():
        head, _, rest = line.partition(":")
        frames.append(int(head))
        balls.append(rest.rpartition(":")[2])
    ball = pd.read_csv(io.StringIO(chr(10).join(balls)), header=None, dtype=np.float32).values
    if not np.array_equal(np.asarray(frames), np.arange(1, len(frames) + 1)):
        raise ValueError("broadcast ATD frame ids are not 1..N")
    return ball


def _load_broadcast():
    import pandas as pd
    from games import require
    from games import GAMES, GAME
    meta, raw, hom = require(["atd_meta", "atd_raw", "homography"], source="broadcast")
    from ..data import CACHE_DIR
    cache = os.path.join(CACHE_DIR, "BAR-ATH_broadcast_ball.npz")
    if os.path.isfile(cache):
        ball = np.load(cache)["ball"]
    else:
        ball = _parse_ball(raw)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, ball=ball)
    h = pd.read_csv(hom, usecols=["frame", "reprojection_error", "valid"])
    n = len(ball)
    valid = np.zeros(n, bool)
    err = np.full(n, np.inf)
    f = h["frame"].to_numpy()
    keep = f < n
    valid[f[keep]] = h["valid"].to_numpy()[keep] == 1
    err[f[keep]] = h["reprojection_error"].to_numpy()[keep]
    length, width = GAMES[GAME]["pitch"]
    return ball[:, :2] * np.array([length, width], np.float32), valid, err


def _second_means(x, step):
    n = len(x) // step
    v = x[:n * step].reshape(n, step, 2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, axis=1)


def _lag(tball, bball, first, last):
    """Tactical frame t matches broadcast frame t - lag, over the tactical half [first, last]."""
    tb = tball[first:last + 1]
    ts = _second_means(tb, COARSE)
    bs = _second_means(bball, COARSE)
    best = None
    for k in range(-len(ts), len(bs)):
        # tactical second i  <->  broadcast second i + k
        lo, hi = max(0, -k), min(len(ts), len(bs) - k)
        if hi - lo < 300:
            continue
        d = np.hypot(*(ts[lo:hi] - bs[lo + k:hi + k]).T)
        ok = ~np.isnan(d)
        if ok.sum() < 300:
            continue
        m = np.median(d[ok])
        if best is None or m < best[0]:
            best = (m, k)
    # tactical second i starts at frame first + 25 i, broadcast second i + k at 25 (i + k):
    # broadcast frame = tactical frame - lag with lag = first - 25 k
    lag0 = first - best[1] * COARSE
    fine = None
    t = np.arange(first, last + 1)
    for lag in range(lag0 - 2 * COARSE, lag0 + 2 * COARSE + 1):
        bi = t - lag
        ok = (bi >= 0) & (bi < len(bball))
        d = np.hypot(*(tball[t[ok]] - bball[bi[ok]]).T)
        good = ~np.isnan(d)
        if good.sum() < 1000:
            continue
        m = np.median(d[good])
        if fine is None or m < fine[0]:
            fine = (m, lag, int(good.sum()))
    return fine


def align(clip):
    """dict(lags, median_m, filled_ball (T, 2), cut (T,) float 0/1/NaN, filled count)."""
    bball, valid, err = _load_broadcast()
    out_ball = clip.ball.copy()
    cut = np.full(clip.n, np.nan, np.float32)
    lags, med, filled = {}, {}, 0
    # broadcast ball usable only where its pitch transform is trustworthy and it does not jump
    trust = valid & (err < MAX_REPROJ)
    jump = np.r_[np.inf, np.hypot(*np.diff(bball, axis=0).T)]
    bclean = np.where((trust & ~(jump > MAX_JUMP_M))[:, None], bball, np.nan)
    for h, a, b in clip.halves:
        m, lag, n_pairs = _lag(clip.ball, bclean, a, b)
        # tactical frame t = broadcast frame t - lag  ==>  broadcast frame = t - lag
        lags[h], med[h] = int(lag), float(m)
        t = np.arange(a, b + 1)
        bi = t - lag
        ok = (bi >= 0) & (bi < len(bball))
        cut[t[ok]] = (~valid[bi[ok]]).astype(np.float32)
        miss = ok & np.isnan(clip.ball[t, 0]) & ~np.isnan(bclean[np.clip(bi, 0, len(bball) - 1), 0])
        out_ball[t[miss]] = bclean[bi[miss]]
        filled += int(miss.sum())
    return dict(lags=lags, median_m=med, ball=out_ball, cut=cut, filled=filled)
