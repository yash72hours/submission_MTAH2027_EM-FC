"""One Clip format for every match: BAR-ATH (inference only) and the public external matches.

Coordinates are metres with the origin at a pitch corner, x along the length. Arrays are indexed by
frame at 25 fps; `halves` lists (half, first, last) inclusive, each half contiguous.

External sources (all public, all gitignored under data/external/, none redistributed):

- DFL / IDSSE open data (Bassek et al. 2025, CC-BY 4.0): 7 Bundesliga matches, optical tracking
  with a per-frame BallStatus (alive/dead) and BallPossession (home/guest).
- Metrica Sports sample games 1-3: Metrica's own tracking; labels from its event data.
- SkillCorner open data (MIT): 20 A-League matches of broadcast tracking at 10 fps with a per-frame
  team in possession; used only as an untouched check, never for training.
"""

import glob
import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL = os.path.join(ROOT, "data", "external")
CACHE = os.path.join(EXTERNAL, "cache")
FPS = 25
PITCH = (105.0, 68.0)


@dataclass
class Clip:
    name: str
    players: np.ndarray        # (T, P, 2) float32 metres, NaN where untracked
    teams: np.ndarray          # (T, P) int8: 0 home, 1 away, -1 not a player at this frame
    keeper: np.ndarray         # (P,) bool
    ball: np.ndarray           # (T, 2) float32 metres, NaN where unseen
    halves: list               # [(half, first, last)]
    home_right: dict           # half -> home attacks towards x = length
    labels: np.ndarray = None  # (T,) '<U1' H/A/D/X, None for BAR-ATH
    pitch: tuple = PITCH
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.ball)


def _keeper_side(players, teams, keeper, first, last, length):
    """True if the home keeper stands on the low-x side over the half (home attacks right)."""
    sl = slice(first, last + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_k = keeper & (np.nanmax(teams[sl] == 0, axis=0) > 0)
        x = np.nanmean(players[sl][:, home_k, 0]) if home_k.any() else np.nan
    if np.isnan(x):   # no keeper seen: use the home team's mean x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x = np.nanmean(np.where(teams[sl] == 0, players[sl][..., 0], np.nan))
    return bool(x < length / 2)


# ----------------------------------------------------------------------------------- BAR-ATH
def barath_clip(use_broadcast=False):
    """BAR-ATH tactical tracking as a Clip (no labels). Keeper slots take the team of the goal
    they guard (data.Match.team_by_frame). With `use_broadcast`, missing tactical ball frames are
    filled from the aligned broadcast tracking and extra["cut"] marks broadcast cut-aways."""
    from ..data import GK_SLOTS, load_match
    m = load_match()
    teams = m.team_by_frame(True).astype(np.int8)
    keeper = np.zeros(m.players.shape[1], bool)
    keeper[list(GK_SLOTS)] = True
    home_right = {1: bool(m.home_attacks_right_first_half), 2: not m.home_attacks_right_first_half}
    clip = Clip("BAR-ATH", m.players.astype(np.float32), teams, keeper, m.ball.astype(np.float32),
                [tuple(h) for h in m.halves], home_right, None, m.pitch, "barath")
    if use_broadcast:
        from . import broadcast
        r = broadcast.align(clip)
        clip.ball = r["ball"].astype(np.float32)
        clip.extra.update(cut=r["cut"], lags=r["lags"], sync_median_m=r["median_m"], filled=r["filled"])
    return clip


# ----------------------------------------------------------------------------------- DFL
DFL_DIR = os.path.join(EXTERNAL, "idsse")
DFL_MATCHES = ("DFL-COM-000001_DFL-MAT-J03WMX", "DFL-COM-000001_DFL-MAT-J03WN1",
               "DFL-COM-000002_DFL-MAT-J03WOH", "DFL-COM-000002_DFL-MAT-J03WOY",
               "DFL-COM-000002_DFL-MAT-J03WPY", "DFL-COM-000002_DFL-MAT-J03WQQ",
               "DFL-COM-000002_DFL-MAT-J03WR9")
_FS = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FR = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')
_BS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def _dfl_info(match):
    from xml.etree import ElementTree as ET
    root = ET.parse(os.path.join(DFL_DIR, "DFL_02_01_matchinformation_{}.xml".format(match))).getroot()
    teams, keepers = {}, set()
    for team in root.iter("Team"):
        teams[team.get("TeamId")] = 0 if team.get("Role") == "home" else 1
        for p in team.iter("Player"):
            if p.get("PlayingPosition") == "TW":
                keepers.add(p.get("PersonId"))
    return teams, keepers


def _parse_dfl(match):
    """{(section, team, person): (N array, X array, Y array, ballposs, ballstatus or None)}"""
    path = os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_{}.xml".format(match))
    sets, cur, buf = {}, None, None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = _FR.match(line)
                buf[0].append(int(m.group(1)))
                buf[1].append(float(m.group(2)))
                buf[2].append(float(m.group(3)))
                if cur[1] == "BALL":
                    b = _BS.search(line)
                    buf[3].append(int(b.group(1)))
                    buf[4].append(int(b.group(2)))
            elif line.startswith("<FrameSet"):
                m = _FS.match(line)
                cur = (m.group(1), m.group(2), m.group(3))
                buf = ([], [], [], [], [])
                sets[cur] = buf
    return {k: tuple(np.asarray(v) for v in b) for k, b in sets.items()}


def dfl_clip(match):
    """A DFL/IDSSE match as a Clip with labels D (BallStatus 0), H/A (BallPossession 1/2)."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "dfl_{}.npz".format(match[-6:]))
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return Clip("DFL-" + match[-6:], z["players"], z["teams"], z["keeper"], z["ball"],
                    [tuple(h) for h in meta["halves"]], {int(k): v for k, v in meta["home_right"].items()},
                    z["labels"], PITCH, "dfl")
    team_of, keepers = _dfl_info(match)
    sets = _parse_dfl(match)
    sections = ("firstHalf", "secondHalf")
    persons = sorted({k[2] for k in sets if k[1] in team_of})
    slot = {p: i for i, p in enumerate(persons)}
    P = len(persons)
    halves, offsets, total = [], {}, 0
    for h, sec in enumerate(sections, 1):
        n_ball = sets[(sec, "BALL", next(k[2] for k in sets if k[0] == sec and k[1] == "BALL"))][0]
        n0, n1 = int(n_ball.min()), int(n_ball.max())
        offsets[sec] = (n0, total)
        halves.append((h, total, total + n1 - n0))
        total += n1 - n0 + 1
    players = np.full((total, P, 2), np.nan, np.float32)
    teams = np.full((total, P), -1, np.int8)
    ball = np.full((total, 2), np.nan, np.float32)
    labels = np.full(total, "X", dtype="<U1")
    keeper = np.array([p in keepers for p in persons])
    for (sec, team, person), (N, X, Y, poss, status) in sets.items():
        n0, base = offsets[sec]
        idx = N - n0 + base
        xy = np.stack([X + PITCH[0] / 2, Y + PITCH[1] / 2], axis=1).astype(np.float32)
        if team == "BALL":
            ball[idx] = xy
            labels[idx] = np.where(status == 0, "D", np.where(poss == 1, "H", "A"))
        elif team in team_of:
            players[idx, slot[person]] = xy
            teams[idx, slot[person]] = team_of[team]
    home_right = {h: _keeper_side(players, teams, keeper, a, b, PITCH[0]) for h, a, b in halves}
    np.savez_compressed(cache, players=players, teams=teams, keeper=keeper, ball=ball, labels=labels,
                        meta=np.array(json.dumps(dict(halves=halves, home_right=home_right))))
    return Clip("DFL-" + match[-6:], players, teams, keeper, ball, halves, home_right, labels, PITCH, "dfl")


# ----------------------------------------------------------------------------------- Metrica
def metrica_clip(game):
    """Metrica sample games 1/2 (CSV) as a Clip, labels from events (possession/external.py)."""
    from .. import external
    m = external.load_metrica(game)
    labels = external.metrica_labels(game, m.n_frames, m.halves)
    keeper = np.zeros(m.players.shape[1], bool)
    keeper[[23, 45]] = True
    teams = np.where(np.isnan(m.players[..., 0]), -1, m.team[None, :]).astype(np.int8)
    home_right = {1: bool(m.home_attacks_right_first_half), 2: not m.home_attacks_right_first_half}
    return Clip("Metrica-{}".format(game), m.players.astype(np.float32), teams, keeper,
                m.ball.astype(np.float32), [tuple(h) for h in m.halves], home_right, labels, m.pitch, "metrica")


M3_DIR = os.path.join(EXTERNAL, "metrica_e706dd506b360d69d9d123d5b8026e7294b13996", "Sample_Game_3")


def metrica3_clip():
    """Metrica sample game 3 (FIFA EPTS text + JSON events) as a Clip; same label rules as 1-2."""
    from xml.etree import ElementTree as ET
    from .. import external
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "metrica3.npz")
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return Clip("Metrica-3", z["players"], z["teams"], z["keeper"], z["ball"],
                    [tuple(h) for h in meta["halves"]], {int(k): v for k, v in meta["home_right"].items()},
                    z["labels"], PITCH, "metrica")
    root = ET.parse(os.path.join(M3_DIR, "Sample_Game_3_metadata.xml")).getroot()
    teams_xml = [t.get("id") for t in root.iter("Team")]
    player_team, keepers = {}, set()
    for p in root.iter("Player"):
        player_team[p.get("id")] = 0 if p.get("teamId") == teams_xml[0] else 1
        for prm in p.iter("ProviderParameter"):
            if prm.findtext("Name") == "position_type" and prm.findtext("Value") == "Goalkeeper":
                keepers.add(p.get("id"))
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [channel_player[r] for r in refs[0::2]]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    params = {prm.findtext("Name"): prm.findtext("Value") for prm in root.find("Metadata").find("GlobalConfig").iter("ProviderParameter")}
    h1 = (int(params["first_half_start"]), int(params["first_half_end"]))
    h2 = (int(params["second_half_start"]), int(params["second_half_end"]))
    pids = sorted(player_team)
    slot = {p: i for i, p in enumerate(pids)}
    n = h2[1] + 1
    players = np.full((n, len(pids), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    scale = np.array(PITCH, np.float32)
    with open(os.path.join(M3_DIR, "Sample_Game_3_tracking.txt")) as fh:
        for line in fh:
            fr, pl, bl = line.rstrip("\n").split(":")
            f = int(fr)
            if f >= n:
                continue
            order = next(o for a, b, o in specs if a <= f <= b)
            vals = np.array([v.split(",") for v in pl.split(";")], dtype=np.float32)
            players[f, [slot[p] for p in order]] = vals * scale
            bv = np.array(bl.split(","), dtype=np.float32)
            ball[f] = bv * scale
    tarr = np.array([player_team[p] for p in pids], np.int8)
    teams = np.where(np.isnan(players[..., 0]), -1, tarr[None, :]).astype(np.int8)
    keeper = np.array([p in keepers for p in pids])
    halves = [(1, h1[0], h1[1]), (2, h2[0], h2[1])]
    ev = json.load(open(os.path.join(M3_DIR, "Sample_Game_3_events.json"), encoding="utf-8"))["data"]
    rows = [dict(Team="Home" if (e.get("team") or {}).get("id") == teams_xml[0] else "Away",
                 Type=e["type"]["name"],
                 Subtype=" ".join(s["name"] for s in (e["subtypes"] if isinstance(e.get("subtypes"), list)
                                                     else [e["subtypes"]] if e.get("subtypes") else [])),
                 Period=e["period"], **{"Start Frame": e["start"]["frame"], "End Frame": e["end"]["frame"] or 0})
            for e in ev]
    labels = external.labels_from_events(pd.DataFrame(rows), n, halves)
    home_right = {h: _keeper_side(players, teams, keeper, a, b, PITCH[0]) for h, a, b in halves}
    np.savez_compressed(cache, players=players, teams=teams, keeper=keeper, ball=ball, labels=labels,
                        meta=np.array(json.dumps(dict(halves=halves, home_right=home_right))))
    return Clip("Metrica-3", players, teams, keeper, ball, halves, home_right, labels, PITCH, "metrica")


# ----------------------------------------------------------------------------------- SkillCorner
SC_DIR = os.path.join(EXTERNAL, "skillcorner")


def skillcorner_ids():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(SC_DIR, "[0-9]*"))
                  if glob.glob(os.path.join(p, "*_tracking_extrapolated.jsonl")))


def skillcorner_clip(mid):
    """A SkillCorner match resampled from 10 to 25 fps (positions linear, labels nearest). Labels are
    H/A where SkillCorner names a team in possession and X elsewhere: it has no dead-ball state, so
    this is a check of the team call and the segmentation of live play only."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "sc_{}.npz".format(mid))
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return Clip("SC-" + mid, z["players"], z["teams"], z["keeper"], z["ball"],
                    [tuple(h) for h in meta["halves"]], {int(k): v for k, v in meta["home_right"].items()},
                    z["labels"], tuple(meta["pitch"]), "skillcorner")
    d = os.path.join(SC_DIR, mid)
    info = json.load(open(os.path.join(d, "{}_match.json".format(mid)), encoding="utf-8"))
    home_id = info["home_team"]["id"]
    length, width = float(info.get("pitch_length") or 105), float(info.get("pitch_width") or 68)
    team_of, keeper_ids = {}, set()
    for p in info["players"]:
        team_of[p["id"]] = 0 if p["team_id"] == home_id else 1
        role = (p.get("player_role") or {}).get("acronym", "") or ""
        if role.upper() == "GK":
            keeper_ids.add(p["id"])
    pids = sorted(team_of)
    slot = {p: i for i, p in enumerate(pids)}
    rows = []
    with open(os.path.join(d, "{}_tracking_extrapolated.jsonl".format(mid)), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("period") not in (1, 2):
                continue
            rows.append(r)
    halves, pieces, total = [], [], 0
    for h in (1, 2):
        rs = [r for r in rows if r["period"] == h]
        f0, f1 = rs[0]["frame"], rs[-1]["frame"]
        n10 = f1 - f0 + 1
        P = np.full((n10, len(pids), 2), np.nan, np.float32)
        B = np.full((n10, 2), np.nan, np.float32)
        L = np.full(n10, "X", dtype="<U1")
        for r in rs:
            i = r["frame"] - f0
            for pd_ in r.get("player_data") or []:
                if pd_["player_id"] in slot:
                    P[i, slot[pd_["player_id"]]] = (pd_["x"] + length / 2, pd_["y"] + width / 2)
            bd = r.get("ball_data") or {}
            if bd.get("x") is not None and bd.get("is_detected", True):
                B[i] = (bd["x"] + length / 2, bd["y"] + width / 2)
            g = (r.get("possession") or {}).get("group")
            if g in ("home team", "away team"):
                L[i] = "H" if g == "home team" else "A"
        # resample 10 -> 25 fps
        n25 = int((n10 - 1) * 2.5) + 1
        t25 = np.arange(n25) / 2.5
        P25 = np.full((n25, len(pids), 2), np.nan, np.float32)
        for j in range(len(pids)):
            for a in range(2):
                v = P[:, j, a]
                ok = ~np.isnan(v)
                if ok.sum() > 1:
                    x = np.interp(t25, np.flatnonzero(ok), v[ok])
                    near = np.interp(t25, np.arange(n10), ok.astype(float)) > 0.99
                    P25[:, j, a] = np.where(near, x, np.nan)
        B25 = np.full((n25, 2), np.nan, np.float32)
        okb = ~np.isnan(B[:, 0])
        for a in range(2):
            if okb.sum() > 1:
                x = np.interp(t25, np.flatnonzero(okb), B[okb, a])
                near = np.interp(t25, np.arange(n10), okb.astype(float)) > 0.99
                B25[:, a] = np.where(near, x, np.nan)
        L25 = L[np.minimum(np.rint(t25).astype(int), n10 - 1)]
        pieces.append((P25, B25, L25))
        halves.append((h, total, total + n25 - 1))
        total += n25
    players = np.concatenate([p[0] for p in pieces])
    ball = np.concatenate([p[1] for p in pieces])
    labels = np.concatenate([p[2] for p in pieces])
    tarr = np.array([team_of[p] for p in pids], np.int8)
    teams = np.where(np.isnan(players[..., 0]), -1, tarr[None, :]).astype(np.int8)
    keeper = np.array([p in keeper_ids for p in pids])
    home_right = {h: _keeper_side(players, teams, keeper, a, b, length) for h, a, b in halves}
    np.savez_compressed(cache, players=players, teams=teams, keeper=keeper, ball=ball, labels=labels,
                        meta=np.array(json.dumps(dict(halves=halves, home_right=home_right, pitch=[length, width]))))
    return Clip("SC-" + mid, players, teams, keeper, ball, halves, home_right, labels, (length, width), "skillcorner")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()
