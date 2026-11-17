"""Make clean external tracking look like BAR-ATH's automatic tracking.

The statistics come from BAR-ATH's *tracking* alone (never its labels or video), measured over its
graded frames:

- the ball-visibility mask: a contiguous stretch of BAR-ATH's own mask, from a random offset,
  replaces the external ball's visibility. It is copied independently of the external labels, so
  a model cannot learn "ball missing means dead ball" from it (how strongly that holds is specific
  to each tracker);
- the number of tracked players per team per frame: players are dropped to match it, choosing
  whom to drop afresh every DROP_BLOCK seconds;
- a slowly varying position bias per player (POS_BIAS_M, re-drawn every BIAS_STEP seconds and
  interpolated, so speeds stay realistic) and white ball noise (BALL_NOISE_M).

The three noise sizes are fixed a priori (typical optical-tracking error is a few decimetres); none
is tuned on any score.
"""

import numpy as np

DROP_BLOCK = 10.0
POS_BIAS_M = 0.3
BIAS_STEP = 2.0
BALL_NOISE_M = 0.2


def barath_stats(clip):
    """(ball_visible, home_count, away_count) over BAR-ATH's graded frames, halves concatenated."""
    vis, hc, ac = [], [], []
    for _, a, b in clip.halves:
        sl = slice(a, b + 1)
        tracked = ~np.isnan(clip.players[sl, :, 0])
        vis.append(~np.isnan(clip.ball[sl, 0]))
        hc.append((tracked & (clip.teams[sl] == 0)).sum(1))
        ac.append((tracked & (clip.teams[sl] == 1)).sum(1))
    return np.concatenate(vis), np.concatenate(hc), np.concatenate(ac)


def _smooth_bias(n, k, fps, rng):
    step = max(1, int(BIAS_STEP * fps))
    knots = np.arange(0, n + step, step)
    vals = rng.normal(0.0, POS_BIAS_M, size=(len(knots), k, 2))
    t = np.arange(n)
    out = np.empty((n, k, 2), np.float32)
    for j in range(k):
        for a in range(2):
            out[:, j, a] = np.interp(t, knots, vals[:, j, a])
    return out


def degrade(clip, stats, seed, ball=True, players=True, fps=25):
    """A degraded copy of `clip` (labels untouched)."""
    import copy
    rng = np.random.default_rng(seed)
    vis_b, hc_b, ac_b = stats
    out = copy.copy(clip)
    P = clip.players.copy()
    B = clip.ball.copy()
    T = clip.teams.copy()
    for _, a, b in clip.halves:
        n = b + 1 - a
        sl = slice(a, b + 1)
        if ball:
            o = int(rng.integers(0, max(1, len(vis_b) - n)))
            mask = np.resize(vis_b[o:o + n], n)
            seg = B[sl]
            seg += rng.normal(0.0, BALL_NOISE_M, seg.shape).astype(np.float32)
            seg[~mask] = np.nan
            B[sl] = seg
        if players:
            o2 = int(rng.integers(0, max(1, len(hc_b) - n)))
            target = {0: np.resize(hc_b[o2:o2 + n], n), 1: np.resize(ac_b[o2:o2 + n], n)}
            seg = P[sl] + _smooth_bias(n, P.shape[1], fps, rng)
            tseg = T[sl]
            block = max(1, int(DROP_BLOCK * fps))
            for s in range(0, n, block):
                e = min(n, s + block)
                prio = rng.random(P.shape[1])
                for team in (0, 1):
                    present = (tseg[s:e] == team) & ~np.isnan(seg[s:e, :, 0])
                    # rank present players by this block's priority; keep the top `target` per frame
                    order = np.argsort(-np.where(present, prio[None, :], -1.0), axis=1, kind="stable")
                    rank = np.empty_like(order)
                    np.put_along_axis(rank, order, np.arange(P.shape[1])[None, :].repeat(e - s, 0), axis=1)
                    keep = rank < target[team][s:e, None]
                    drop = present & ~keep
                    sub = seg[s:e]
                    sub[drop] = np.nan
                    seg[s:e] = sub
            P[sl] = seg
    out.players, out.ball, out.teams = P, B, T
    out.name = clip.name + "~deg{}".format(seed)
    return out
