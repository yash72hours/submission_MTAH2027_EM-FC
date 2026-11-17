"""LightGBM frame model on external matches, the min-stoppage Viterbi decoder, BAR-ATH inference."""

import json
import os

import numpy as np

from . import degrade, features, labels as lab, sources

CODE = {"H": 0, "A": 1, "D": 2}
LGB_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, deterministic=True, force_row_wise=True,
                  num_threads=12, seed=0)
ROUNDS = 300
TRAIN_STEP = 5          # train on every 5th frame (5 fps): neighbouring frames are near-copies
FEATURE_CACHE = os.path.join(sources.CACHE, "features")


MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_manifest.json")


def dfl_complete(match):
    """True when all three DFL files of `match` are present at their pinned sizes."""
    files = json.load(open(MANIFEST))["idsse"]["files"]
    for k in ("02_01_matchinformation", "03_02_events_raw", "04_03_positions_raw_observed"):
        name = "DFL_{}_{}.xml".format(k, match)
        path = os.path.join(sources.DFL_DIR, name)
        if not (os.path.isfile(path) and os.path.getsize(path) == files[name]["size"]):
            return False
    return True


def training_clips():
    """The external matches used for training and leave-one-match-out validation."""
    return ([("dfl", m) for m in sources.DFL_MATCHES if dfl_complete(m)]
            + [("metrica", 1), ("metrica", 2), ("metrica", 3)])


def clip_seed(kind, key):
    """A fixed degradation seed per match (independent of which matches are present)."""
    if kind == "dfl":
        return sources.DFL_MATCHES.index(key)
    if kind == "metrica":
        return 10 + int(key)
    return 100 + int(key) % 1000


def load_clip(kind, key):
    if kind == "dfl":
        return sources.dfl_clip(key)
    if kind == "metrica":
        return sources.metrica3_clip() if key == 3 else sources.metrica_clip(key)
    if kind == "skillcorner":
        return sources.skillcorner_clip(key)
    raise ValueError(kind)


def prepared(kind, key, stats, seed):
    """(name, names, X, y, halves) for one external match: degraded (DFL: ball and players;
    Metrica, same tracker as BAR-ATH: players only, its own ball gaps kept; SkillCorner: as is),
    labels aligned to controlled possession, features computed. Cached on disk."""
    os.makedirs(FEATURE_CACHE, exist_ok=True)
    tag = "{}_{}_s{}".format(kind, str(key)[-6:], seed)
    path = os.path.join(FEATURE_CACHE, tag + ".npz")
    if os.path.isfile(path):
        z = np.load(path, allow_pickle=False)
        return tag, list(z["names"]), z["X"], z["y"], [tuple(h) for h in z["halves"]]
    clip = load_clip(kind, key)
    if kind == "dfl":
        clip = degrade.degrade(clip, stats, seed, ball=True, players=True)
    elif kind == "metrica":
        clip = degrade.degrade(clip, stats, seed, ball=False, players=True)
    y = lab.controlled(clip.labels)
    names, X = features.compute(clip)
    np.savez(path, names=np.array(names), X=X, y=y, halves=np.array(clip.halves))
    return tag, names, X, y, clip.halves


def rows(X, y, names, step=TRAIN_STEP, swap=True):
    """Training rows (every `step`-th labelled frame), with the home/away swap appended."""
    idx = np.flatnonzero(np.isin(y, ("H", "A", "D")))
    idx = idx[idx % step == 0]
    Xr, yr = X[idx], np.array([CODE[v] for v in y[idx]], np.int8)
    if not swap:
        return Xr, yr
    si = features.swap_index(names)
    ys = np.where(yr == 0, 1, np.where(yr == 1, 0, yr)).astype(np.int8)
    return np.vstack([Xr, Xr[:, si]]), np.r_[yr, ys]


def train(datasets, rounds=ROUNDS, params=None):
    import lightgbm as lgb
    Xs, ys = [], []
    for names, X, y in datasets:
        a, b = rows(X, y, names)
        Xs.append(a)
        ys.append(b)
    data = lgb.Dataset(np.vstack(Xs), label=np.concatenate(ys), feature_name=list(datasets[0][0]),
                       free_raw_data=True)
    return lgb.train(dict(LGB_PARAMS, **(params or {})), data, num_boost_round=rounds)


def predict(booster, X, symmetric=True, names=None):
    """(T, 3) probabilities H/A/D; with `symmetric`, the mean of the model on the match and on the
    match with the teams swapped (H/A exchanged back), so the call cannot favour a side."""
    p = booster.predict(X, num_threads=LGB_PARAMS["num_threads"])
    if symmetric:
        q = booster.predict(X[:, features.swap_index(names)], num_threads=LGB_PARAMS["num_threads"])
        p = 0.5 * (p + q[:, [1, 0, 2]])
    return p.astype(np.float32)


# --------------------------------------------------------------------------- decoding
DEFAULT_DECODER = dict(c_team=8.0, c_dead=6.0, min_dead=25, scale=1.0)


def decode_half(p, setting, evidence=None, boundary=None):
    """H/A/D labels for one half from (n, 3) probabilities: Viterbi over H, A and a dead state
    (two copies, so the decoder's four-state code is reused) with switch costs and a minimum
    stoppage length. `evidence` (n,) is an optional log-likelihood ratio added to the dead state."""
    from ..decode import viterbi_min_dead
    s = setting["scale"] * np.log(np.clip(p, 1e-4, 1.0))
    if evidence is not None:
        s = s.copy()
        s[:, 2] += evidence
    scores = np.c_[s[:, 0], s[:, 1], s[:, 2], s[:, 2]]
    c = np.zeros((4, 4))
    c[0, 1] = c[1, 0] = setting["c_team"]
    c[0, 2:] = c[1, 2:] = setting["c_dead"]
    c[2:, :2] = setting["c_dead"]
    if boundary is not None and setting.get("bnd_a", 0) > 0:
        # a learned possession-boundary probability makes a direct team switch cheaper there
        from ..decode import viterbi_switch
        assert int(setting["min_dead"]) <= 1
        team_cost = setting["c_team"] * (1.0 - setting["bnd_a"] * np.asarray(boundary, np.float64))
        path = viterbi_switch(scores, c, team_cost)
    else:
        path = viterbi_min_dead(scores, c, int(setting["min_dead"]))
    return np.array(["H", "A", "D", "D"])[path]


def decode_clip(p, halves, n, setting, evidence=None, boundary=None):
    out = np.full(n, "X", dtype="<U1")
    for _, a, b in halves:
        ev = None if evidence is None else evidence[a:b + 1]
        bd = None if boundary is None else boundary[a:b + 1]
        out[a:b + 1] = decode_half(p[a:b + 1], setting, ev, bd)
    return out


def spell_scores(truth, pred):
    import sys
    sys.path.insert(0, os.path.join(sources.ROOT, "evaluation"))
    import possession_metrics as pm
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(pred), bootstrap_n=0)
    return dict(spell_f1=r["spell_f1_mean"], macro_f1=r["macro_f1"], n_pred=r.get("n_pred_spells"),
                n_gt=r.get("n_gt_spells"))


# --------------------------------------------------------------------------- match evidence (EM)
GAP_BINS = (0.0, 2.0, 5.0, 10.0, 30.0)   # seconds; category 0 = ball seen
EVIDENCE_WEIGHT = 0.5                     # fixed a priori: half the estimated log-ratio


def gap_category(ball):
    """(T,) int: 0 where the ball is seen, else 1 + bin of the length of the gap it sits in."""
    vis = ~np.isnan(ball[:, 0])
    cat = np.zeros(len(ball), np.int8)
    e = np.diff(np.r_[0, (~vis).astype(np.int8), 0])
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        cat[s:t] = 1 + np.searchsorted(GAP_BINS, (t - s) / 25.0, side="right") - 1
    return cat


def em_evidence(categories, p_dead, graded, weight=EVIDENCE_WEIGHT, floor=1e-3):
    """One EM step without labels: the model's own dead-ball probability gives the expected
    category counts under dead and alive play, and each category's log-ratio is the evidence
    (times `weight`) added to the dead state. Returns (evidence (T,), {category: log-ratio})."""
    c = categories[graded]
    w = p_dead[graded].astype(np.float64)
    cats = np.unique(c)
    pd_ = np.array([w[c == k].sum() for k in cats]) + floor
    pa_ = np.array([(1 - w)[c == k].sum() for k in cats]) + floor
    llr = np.log(pd_ / pd_.sum()) - np.log(pa_ / pa_.sum())
    table = {int(k): float(v) for k, v in zip(cats, llr)}
    ev = np.zeros(len(categories), np.float32)
    for k, v in table.items():
        ev[categories == k] = weight * v
    ev[~graded] = 0
    return ev, table


def save_booster(booster, path):
    booster.save_model(path, num_iteration=-1)


def load_booster(path):
    import lightgbm as lgb
    return lgb.Booster(model_file=path)
