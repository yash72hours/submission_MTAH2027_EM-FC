"""Metrica Sports sample games 1 and 2 (https://github.com/metrica-sports/sample-data, pinned commit
below): loader and possession labels. They are training and leave-one-match-out validation data for
the frame model. Nothing here touches the provided game.

Download (about 127 MB; data/ is gitignored, the files are not redistributed):

    python tools/download_external.py

Labels come from Metrica's own event data by fixed rules, decided before any fitting:

- **Owner.** The team of the latest PASS, RECOVERY, SHOT, BALL LOST or SET PIECE at or before the
  frame. CHALLENGE events never change the owner (a duel or a touch without control is not a
  change of possession), and between a BALL LOST and the next RECOVERY the ball stays with the
  team that lost it (a loose ball stays with the last team that had it).
- **Dead.** From the moment the ball goes out (BALL OUT, a SHOT ending OUT or GOAL: their end
  frame), a foul (FAULT RECEIVED), or an offside/hand-ball loss (their start frame), until the
  start frame of the next SET PIECE.
- **Unknown (X).** Before the first event of a period and after its last.
"""

import hashlib
import os
import warnings

import numpy as np
import pandas as pd

from .data import AWAY, GK_SLOTS, HOME, Match

COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/data".format(COMMIT)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DIR = os.path.join(ROOT, "data", "external", "metrica_" + COMMIT)
KINDS = ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")
SHA256 = {
    (1, "RawTrackingData_Home_Team"): "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
    (1, "RawTrackingData_Away_Team"): "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
    (1, "RawEventsData"): "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
    (2, "RawTrackingData_Home_Team"): "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
    (2, "RawTrackingData_Away_Team"): "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
    (2, "RawEventsData"): "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
}
PITCH = (105.0, 68.0)   # Metrica's documented pitch for normalised coordinates
FPS = 25
MAX_SPEED = 12.0        # m/s; faster position steps are tracking jumps, not running
SPEED_HALF_SPAN = 2     # player speed from a centred difference over +/- 2 frames
OWNER_TYPES = ("PASS", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE", "CARRY")   # CARRY: game 3 only


def relative(game, kind):
    return "Sample_Game_{0}/Sample_Game_{0}_{1}.csv".format(game, kind)


def path_of(game, kind, root=EXTERNAL_DIR):
    return os.path.join(root, relative(game, kind))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 23), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(root=EXTERNAL_DIR):
    """Fetch the six pinned files and check their hashes."""
    import urllib.request
    for (game, kind), expected in SHA256.items():
        path = path_of(game, kind, root)
        if not (os.path.isfile(path) and sha256(path) == expected):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve("{}/{}".format(BASE_URL, relative(game, kind)), path)
        if sha256(path) != expected:
            raise ValueError("{}: hash differs from the pinned Metrica file".format(path))


def check(game, root=EXTERNAL_DIR):
    for kind in KINDS:
        path = path_of(game, kind, root)
        if not os.path.isfile(path):
            raise FileNotFoundError("{} missing; run: python tools/fit_external.py download".format(path))
        if sha256(path) != SHA256[(game, kind)]:
            raise ValueError("{}: hash differs from the pinned Metrica file".format(path))


def _team_table(path):
    table = pd.read_csv(path, skiprows=2)
    players = [c for c in table.columns if str(c).startswith("Player")]
    columns = list(table.columns)
    xy = np.stack([table.iloc[:, [columns.index(c), columns.index(c) + 1]].to_numpy(np.float64)
                   for c in players], axis=1)
    ball = table.iloc[:, -2:].to_numpy(np.float64)
    return table, xy, ball


def _speed(xy, halves, fps):
    """(T, P) m/s from a centred difference within each half; jumps above MAX_SPEED are NaN."""
    out = np.full(xy.shape[:2], np.nan)
    k = SPEED_HALF_SPAN
    for _, first, last in halves:
        seg = xy[first:last + 1]
        d = seg[2 * k:] - seg[:-2 * k]
        out[first + k:last + 1 - k] = np.hypot(d[..., 0], d[..., 1]) * fps / (2 * k)
    out[out > MAX_SPEED] = np.nan
    return out


def load_metrica(game, root=EXTERNAL_DIR):
    """A Match on Metrica's frame index (row i = Frame i; row 0 is empty). Keepers go to the
    GK_SLOTS so the signals treat them as keepers; each keeper is the player furthest from the
    halfway line on average."""
    check(game, root)
    home, hxy, hball = _team_table(path_of(game, "RawTrackingData_Home_Team", root))
    away, axy, aball = _team_table(path_of(game, "RawTrackingData_Away_Team", root))
    if not np.array_equal(home["Frame"], away["Frame"]):
        raise ValueError("Metrica home/away frame indices differ")
    if not np.allclose(hball, aball, equal_nan=True):
        raise ValueError("Metrica home/away ball tracks differ")
    frames = home["Frame"].to_numpy(int)
    n = int(frames[-1]) + 1
    period = np.zeros(n, dtype=np.int8)
    period[frames] = home["Period"].to_numpy(np.int8)
    halves = []
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        halves.append((half, int(idx[0]), int(idx[-1])))

    scale = np.array(PITCH)
    players = np.full((n, 46, 2), np.nan)
    team = np.r_[np.full(24, HOME), np.full(22, AWAY)].astype(np.int8)
    keepers = {}
    for side, xy, slots, gk in ((HOME, hxy, list(range(23)), GK_SLOTS[0]),
                                (AWAY, axy, list(range(24, 45)), GK_SLOTS[1])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            keeper = int(np.nanargmax(np.nanmean(np.abs(xy[:, :, 0] - 0.5), axis=0)))
        if xy.shape[1] > len(slots) + 1:
            raise ValueError("more Metrica players than slots")
        for k in range(xy.shape[1]):
            slot = gk if k == keeper else slots.pop(0)
            players[frames, slot] = xy[:, k] * scale
        keepers[side] = keeper
    ball = np.full((n, 2), np.nan)
    ball[frames] = hball * scale

    # Home attacks right in the first half if its keeper stands on the low-x side.
    first = halves[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_keeper_x = np.nanmean(players[first[1]:first[2] + 1, GK_SLOTS[0], 0])

    speed = _speed(players, halves, FPS)
    return Match(players=players, player_speed=speed, team=team, ball=ball, homography=None,
                 pitch=PITCH, fps=FPS, halves=halves,
                 home_attacks_right_first_half=bool(home_keeper_x < PITCH[0] / 2))


def empty_tags():
    return pd.DataFrame({"code": pd.Series(dtype=object), "start_frame": pd.Series(dtype=int),
                         "end_frame": pd.Series(dtype=int), "team": pd.Series(dtype=int),
                         "type": pd.Series(dtype=object), "half": pd.Series(dtype=object)})


def metrica_labels(game, n_frames, halves, root=EXTERNAL_DIR):
    """(T,) 'H' / 'A' / 'D' / 'X' from Metrica's events by the rules in the module docstring."""
    return labels_from_events(pd.read_csv(path_of(game, "RawEventsData", root)), n_frames, halves)


def labels_from_events(events, n_frames, halves):
    """The same rules for an events table with Metrica's CSV columns (Team Home/Away, Type,
    Subtype, Period, Start Frame, End Frame)."""
    events = events[["Team", "Type", "Subtype", "Period", "Start Frame", "End Frame"]].rename(
        columns={"Start Frame": "StartFrame", "End Frame": "EndFrame"})
    events["Subtype"] = events["Subtype"].fillna("").astype(str).str.upper()
    events = events.sort_values(["StartFrame", "EndFrame"], kind="stable").reset_index(drop=True)
    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in halves:
        ev = events[events["Period"] == half]
        if not len(ev):
            continue
        owner = np.full(n_frames, "", dtype="<U1")
        dead = np.zeros(n_frames, dtype=bool)
        # owner marks: the latest owner event at or before each frame
        marks = np.full(last + 2 - first, "", dtype="<U1")
        for row in ev.itertuples(index=False):
            if row.Type in OWNER_TYPES and row.Team in ("Home", "Away"):
                start = int(row.StartFrame)
                if first <= start <= last:
                    marks[start - first] = "H" if row.Team == "Home" else "A"
        current = ""
        for i in range(len(marks) - 1):
            if marks[i]:
                current = marks[i]
            owner[first + i] = current
        # dead stretches: stoppage onset until the next SET PIECE start
        set_starts = np.sort(ev.loc[ev["Type"] == "SET PIECE", "StartFrame"].to_numpy(int))
        for row in ev.itertuples(index=False):
            start, end = int(row.StartFrame), int(row.EndFrame)
            onset = None
            if row.Type == "BALL OUT":
                onset = end if end > 0 else start
            elif row.Type == "SHOT" and ("OUT" in row.Subtype or "GOAL" in row.Subtype):
                onset = end if end > 0 else start
            elif row.Type == "FAULT RECEIVED":
                onset = start
            elif row.Type == "BALL LOST" and ("OFFSIDE" in row.Subtype or "HAND BALL" in row.Subtype):
                onset = start
            if onset is None:
                continue
            later = set_starts[set_starts > onset]
            stop = int(later[0]) if len(later) else last + 1
            dead[max(onset, first):min(stop, last + 1)] = True
        lo = int(ev["StartFrame"].min())
        hi = int(max(ev["StartFrame"].max(), ev["EndFrame"].max()))
        span = np.arange(max(lo, first), min(hi, last) + 1)
        lab = np.where(dead[span], "D", owner[span])
        lab = np.where(lab == "", "X", lab)
        labels[span] = lab
    return labels
