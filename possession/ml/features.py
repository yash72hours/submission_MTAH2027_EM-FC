"""Team-level, permutation-invariant features with multi-scale temporal context.

Automatic tracking has no stable identities (slots are tracklets), so every feature describes a
team or the ball, never a player. Team features come in pairs (`h_*` home, `a_*` away), each in
that team's own attacking frame, so swapping the teams is an exact column permutation (`swap`):
that doubles the training data and stops a model from learning that one side usually has the ball.

Speeds are computed from positions for every source (never a provider's own speed channel), so
all trackers are measured the same way.
"""

import numpy as np

SPEED_SPAN = 2           # centred difference over +/- 2 frames
MAX_SPEED = 12.0
CAP_GAP_S = 60.0
CONTEXT = {"c1": (-12, 12), "c3": (-37, 37), "c9": (-112, 112),
           "p2": (-50, -1), "p6": (-150, -1), "f2": (1, 50), "f6": (1, 150)}
KEY = ["g_vis", "g_bspeed", "g_act", "g_still", "g_oob", "g_fast", "g_since", "g_until",
       "h_d1", "a_d1", "h_n2", "a_n2", "h_dd", "a_dd", "h_bx", "a_bx", "h_cx", "a_cx",
       "h_spd", "a_spd", "h_vatt", "a_vatt", "h_run", "a_run"]


def _speed(xy, fps):
    """(n, P) speed and (n, P) x-velocity from a centred difference; NaN beyond MAX_SPEED."""
    k = SPEED_SPAN
    v = np.full(xy.shape, np.nan, np.float32)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (fps / (2 * k))
    s = np.hypot(v[..., 0], v[..., 1])
    bad = s > MAX_SPEED
    s[bad] = np.nan
    vx = v[..., 0].copy()
    vx[bad] = np.nan
    return s, vx


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def _roll(x, lo, hi):
    """NaN-aware mean of x over [t+lo, t+hi] (clipped at the ends)."""
    n = len(x)
    ok = ~np.isnan(x)
    cs = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0), dtype=np.float64)]
    cc = np.r_[0, np.cumsum(ok)]
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi + 1, 0, n)
    cnt = cc[b] - cc[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[b] - cs[a]) / cnt
    out[cnt == 0] = np.nan
    return out.astype(np.float32)


def _nanstat(fn, x, mask):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(np.where(mask, x, np.nan), axis=1).astype(np.float32)


def _half(players, teams, keeper, ball, home_right, pitch, fps):
    L, W = pitch
    n, P = players.shape[:2]
    f = {}
    vis = ~np.isnan(ball[:, 0])
    f["g_vis"] = vis.astype(np.float32)
    bs, _ = _speed(ball[:, None, :], fps)
    f["g_bspeed"] = bs[:, 0]
    # time since / until the ball was seen, and the length of the gap a frame sits in
    since = np.zeros(n, np.float32)
    until = np.zeros(n, np.float32)
    gap = np.zeros(n, np.float32)
    for s, e in zip(*_runs(~vis)):
        ln = (e - s) / fps
        gap[s:e] = ln
        since[s:e] = (np.arange(e - s) + 1) / fps
        until[s:e] = (e - s - np.arange(e - s)) / fps
        if s == 0:
            since[s:e] = CAP_GAP_S
        if e == n:
            until[s:e] = CAP_GAP_S
    f["g_since"] = np.minimum(since, CAP_GAP_S)
    f["g_until"] = np.minimum(until, CAP_GAP_S)
    f["g_gap"] = np.minimum(gap, 2 * CAP_GAP_S)
    bx, by = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        f["g_oob"] = np.where(vis, ((bx < -1) | (bx > L + 1) | (by < -1) | (by > W + 1)), np.nan).astype(np.float32)
        moved = np.full(n, np.nan, np.float32)
        moved[12:-12] = np.hypot(*(ball[24:] - ball[:-24]).T)
        f["g_still"] = np.where(np.isnan(moved), np.nan, moved < 1.0).astype(np.float32)
        f["g_by"] = np.abs(by - W / 2).astype(np.float32)
        f["g_touch"] = np.where(vis, np.minimum(by, W - by) < 1.5, np.nan).astype(np.float32)
        end = np.minimum(bx, L - bx)
        f["g_end"] = end.astype(np.float32)
        f["g_corner"] = np.where(vis, (end < 2.5) & (np.minimum(by, W - by) < 2.5), np.nan).astype(np.float32)
        f["g_centre"] = np.where(vis, np.hypot(bx - L / 2, by - W / 2) < 1.5, np.nan).astype(np.float32)

    tracked = ~np.isnan(players[..., 0])
    speed, vx = _speed(players, fps)
    out_mask = tracked & ~keeper[None, :]
    f["g_act"] = _nanstat(np.nanmean, speed, out_mask & (teams >= 0))
    f["g_fast"] = _nanstat(np.nanmean, (speed > 4).astype(np.float32), out_mask & (teams >= 0) & ~np.isnan(speed))
    f["g_n"] = (tracked & (teams >= 0)).sum(1).astype(np.float32)

    d = np.hypot(players[..., 0] - bx[:, None], players[..., 1] - by[:, None])
    for code, team in (("h", 0), ("a", 1)):
        right = home_right if team == 0 else not home_right
        sgn = 1.0 if right else -1.0
        m = tracked & (teams == team)
        mo = m & ~keeper[None, :]
        dm = np.where(m, d, np.inf)
        two = np.sort(np.partition(dm, 1, axis=1)[:, :2], axis=1) if P > 1 else np.c_[dm, dm]
        d1 = np.where(np.isfinite(two[:, 0]), two[:, 0], np.nan).astype(np.float32)
        d2 = np.where(np.isfinite(two[:, 1]), two[:, 1], np.nan).astype(np.float32)
        f[code + "_d1"], f[code + "_d2"] = d1, d2
        f[code + "_n2"] = np.where(vis, (m & (d < 2)).sum(1), np.nan).astype(np.float32)
        f[code + "_n5"] = np.where(vis, (m & (d < 5)).sum(1), np.nan).astype(np.float32)
        f[code + "_n10"] = np.where(vis, (m & (d < 10)).sum(1), np.nan).astype(np.float32)
        xa = players[..., 0] if right else L - players[..., 0]
        f[code + "_bx"] = (bx if right else L - bx).astype(np.float32)
        f[code + "_cx"] = _nanstat(np.nanmean, xa, mo)
        f[code + "_sx"] = _nanstat(np.nanstd, xa, mo)
        f[code + "_sy"] = _nanstat(np.nanstd, players[..., 1], mo)
        f[code + "_front"] = _nanstat(np.nanmax, xa, mo)
        f[code + "_back"] = _nanstat(np.nanmin, xa, mo)
        f[code + "_spd"] = _nanstat(np.nanmean, speed, mo)
        f[code + "_vatt"] = _nanstat(np.nanmean, sgn * vx, mo)
        f[code + "_run"] = _nanstat(np.nanmean, (speed > 4).astype(np.float32), mo & ~np.isnan(speed))
        f[code + "_walk"] = _nanstat(np.nanmean, (speed < 1).astype(np.float32), mo & ~np.isnan(speed))
        f[code + "_cnt"] = m.sum(1).astype(np.float32)
    f["h_dd"] = f["h_d1"] - f["a_d1"]
    f["a_dd"] = f["a_d1"] - f["h_d1"]
    f["h_cxb"] = f["h_cx"] - f["h_bx"]
    f["a_cxb"] = f["a_cx"] - f["a_bx"]
    for k in KEY:
        for tag, (lo, hi) in CONTEXT.items():
            f["{}_{}".format(k, tag)] = _roll(f[k], lo, hi)
    return f


def compute(clip):
    """(names, X) with X (T, F) float32, NaN outside the halves."""
    parts = {}
    names = None
    X = None
    for h, a, b in clip.halves:
        sl = slice(a, b + 1)
        f = _half(clip.players[sl], clip.teams[sl], clip.keeper, clip.ball[sl],
                  clip.home_right[h], clip.pitch, 25)
        if names is None:
            names = list(f)
            X = np.full((clip.n, len(names)), np.nan, np.float32)
        X[sl] = np.stack([f[k] for k in names], axis=1)
    return names, X


def swap_index(names):
    """Column permutation that exchanges home and away."""
    pos = {k: i for i, k in enumerate(names)}
    out = []
    for k in names:
        if k.startswith("h_"):
            out.append(pos["a_" + k[2:]])
        elif k.startswith("a_"):
            out.append(pos["h_" + k[2:]])
        else:
            out.append(pos[k])
    return np.array(out)
