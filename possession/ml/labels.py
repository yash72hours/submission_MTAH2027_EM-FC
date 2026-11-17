"""Align external possession labels to *controlled* possession.

Provider labels change team on every touch (DFL's BallPossession follows the last player on the
ball). The challenge counts a change of possession only on control: "a simple touch of the ball
without control" and "a deflection of a pass that does not intercept it" do not count, and a
proxy for control is "a player keeps the ball for a few seconds" (CHALLENGE.md).

So a team spell shorter than TAU seconds that ends with the previous team back on the ball, or
with play stopping, is a touch and goes back to the previous team. TAU = 2 s is fixed from that
definition ("a few seconds", taken at its shortest) before any fitting; it is not chosen by any
score, and never by the BAR-ATH ground truth.
"""

import numpy as np

TAU_SECONDS = 2.0


def runs(labels):
    b = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return list(zip(b[:-1], b[1:]))


def controlled(labels, fps=25, tau=TAU_SECONDS):
    """Return a copy of `labels` (H/A/D/X) with touch spells merged back, by the rule above."""
    out = labels.copy()
    k = int(round(tau * fps))
    changed = True
    while changed:
        changed = False
        rr = runs(out)
        for i in range(1, len(rr) - 1):
            a, b = rr[i]
            lab = out[a]
            if lab not in "HA" or b - a >= k:
                continue
            prev = out[rr[i - 1][0]]
            nxt = out[rr[i + 1][0]]
            if prev in "HA" and prev != lab and (nxt == prev or nxt == "D"):
                out[a:b] = prev
                changed = True
        # one pass at a time keeps the rule local and the result order-independent enough
    return out


def rhythm(labels, fps=25):
    """(possessions per live minute, median possession seconds) of a label array."""
    rr = [(a, b) for a, b in runs(labels) if labels[a] in "HA"]
    live = np.isin(labels, ("H", "A")).sum() / fps / 60
    d = np.array([(b - a) / fps for a, b in rr]) if rr else np.array([np.nan])
    return len(rr) / max(live, 1e-9), float(np.median(d))
