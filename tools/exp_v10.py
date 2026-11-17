"""v10 candidates, judged only on the 10 external matches (leave-one-match-out), like v9.

    python tools/exp_v10.py prep2       # a second degradation draw of every training match
    python tools/exp_v10.py lgb2        # LightGBM CV trained on both draws (held-out match: draw 1)
    python tools/exp_v10.py gru_seeds   # GRU CV with extra seeds (draw 1 only, as v9)
    python tools/exp_v10.py blend       # blend/decoder grid over every candidate combination

Nothing here reads BAR-ATH labels or video (RULES.md rule 4). The held-out match is always scored
on its v9 draw, so every number is comparable with v9's 0.708.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from possession.ml import model  # noqa: E402
import train_v9 as t9  # noqa: E402

WORK = os.path.join(t9.WORK, "v10")
DRAW2 = 50              # second draw seed offset (fixed before looking at any result)
GRU_SEEDS = (1, 2)


def prep2():
    st = t9.stats()
    out = []
    for kind, key in model.training_clips():
        t = time.time()
        d = model.prepared(kind, key, st, model.clip_seed(kind, key) + DRAW2)
        out.append(d)
        print("prepared", d[0], d[2].shape, "{:.0f}s".format(time.time() - t), flush=True)
    return out


def _load(tag):
    z = np.load(os.path.join(model.FEATURE_CACHE, tag + ".npz"), allow_pickle=False)
    return list(z["names"]), z["X"], z["y"], [tuple(h) for h in z["halves"]]


def pairs():
    """[(draw-1 tag, draw-2 tag)] per training match."""
    out = []
    for kind, key in model.training_clips():
        s = model.clip_seed(kind, key)
        base = "{}_{}_s".format(kind, str(key)[-6:])
        out.append((base + str(s), base + str(s + DRAW2)))
    return out


def lgb2():
    d = os.path.join(WORK, "oof_lgb2")
    os.makedirs(d, exist_ok=True)
    P = pairs()
    for i, (held, _) in enumerate(P):
        path = os.path.join(d, held + ".npy")
        if os.path.isfile(path):
            continue
        t = time.time()
        import lightgbm as lgb
        Xs, ys = [], []
        for j, (a, b) in enumerate(P):
            if j != i:
                for tag in (a, b):
                    n, X, y, _ = _load(tag)
                    r = model.rows(X, y, n)      # same rows/swap as model.train, loaded one clip at a time
                    Xs.append(r[0].astype(np.float32))
                    ys.append(r[1])
                    del X
        data = lgb.Dataset(np.vstack(Xs), label=np.concatenate(ys), feature_name=list(n), free_raw_data=True)
        del Xs
        booster = lgb.train(dict(model.LGB_PARAMS), data, num_boost_round=model.ROUNDS)
        del data
        n, X, _, _ = _load(held)
        np.save(path, model.predict(booster, X, names=n))
        print("lgb2 fold", held, "{:.0f}s".format(time.time() - t), flush=True)


def _gru_fold(args):
    i, seed = args
    from possession.ml import gru
    tags = [a for a, _ in pairs()]
    held = tags[i]
    data = {t: _load(t) for t in tags}
    names = data[held][0]
    tr = [t for t in tags if t != held]
    sc = gru.scaler([data[t][1] for t in tr])
    seqs = [s for t in tr for s in gru.sequences(data[t][1], data[t][2], data[t][3], sc)]
    w = gru.train(seqs, names, seed=seed, threads=3)
    p = gru.predict(w, sc, data[held][1], data[held][3], names)
    np.save(os.path.join(WORK, "oof_gru_s{}".format(seed), held + ".npy"), p)
    return held, seed


def gru_seeds(jobs):
    from multiprocessing import Pool
    tags = [a for a, _ in pairs()]
    work = []
    for s in GRU_SEEDS:
        os.makedirs(os.path.join(WORK, "oof_gru_s{}".format(s)), exist_ok=True)
        work += [(i, s) for i, t in enumerate(tags)
                 if not os.path.isfile(os.path.join(WORK, "oof_gru_s{}".format(s), t + ".npy"))]
    t0 = time.time()
    with Pool(jobs) as pool:
        for held, s in pool.imap_unordered(_gru_fold, work):
            print("gru fold", held, "seed", s, "{:.0f}s".format(time.time() - t0), flush=True)


# ----------------------------------------------------------------------------- boundary model
BND_WIN = 12            # a direct H<->A switch within +/- 0.5 s (fixed a priori)
BND_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  deterministic=True, force_row_wise=True, num_threads=4, seed=0)
BND_ROUNDS = 200


def boundary_target(y):
    """(T,) 0/1: a direct change of team in the controlled labels within +/- BND_WIN frames."""
    team = np.isin(y, ("H", "A"))
    sw = np.flatnonzero(team[1:] & team[:-1] & (y[1:] != y[:-1])) + 1
    b = np.zeros(len(y), np.int8)
    for t in sw:
        b[max(0, t - BND_WIN):t + BND_WIN + 1] = 1
    return b


def _bnd_rows(names, X, y, step=model.TRAIN_STEP):
    idx = np.flatnonzero(np.isin(y, ("H", "A", "D")))
    idx = idx[idx % step == 0]
    b = boundary_target(y)[idx]
    si = __import__("possession.ml.features", fromlist=["swap_index"]).swap_index(names)
    Xr = X[idx]
    return np.vstack([Xr, Xr[:, si]]), np.r_[b, b]


def bnd_predict(booster, X, names):
    from possession.ml import features
    p = booster.predict(X, num_threads=BND_PARAMS["num_threads"])
    q = booster.predict(X[:, features.swap_index(names)], num_threads=BND_PARAMS["num_threads"])
    return (0.5 * (p + q)).astype(np.float32)


def bnd_cv():
    import lightgbm as lgb
    d = os.path.join(WORK, "oof_bnd")
    os.makedirs(d, exist_ok=True)
    tags = [a for a, _ in pairs()]
    for held in tags:
        path = os.path.join(d, held + ".npy")
        if os.path.isfile(path):
            continue
        t = time.time()
        Xs, ys = [], []
        for tag in tags:
            if tag != held:
                n, X, y, _ = _load(tag)
                a, b = _bnd_rows(n, X, y)
                Xs.append(a)
                ys.append(b)
        data = lgb.Dataset(np.vstack(Xs), label=np.concatenate(ys), feature_name=list(n), free_raw_data=True)
        del Xs
        booster = lgb.train(BND_PARAMS, data, num_boost_round=BND_ROUNDS)
        n, X, y, _ = _load(held)
        pb = bnd_predict(booster, X, n)
        np.save(path, pb)
        tgt = boundary_target(y)
        print("bnd fold", held, "{:.0f}s".format(time.time() - t), "mean p on/off target {:.3f}/{:.3f}".format(
            pb[tgt == 1].mean(), pb[tgt == 0].mean()), flush=True)


# ----------------------------------------------------------------------------- blend
def _probs(tag, lgb_kind, gru_kind):
    lg = {"v9": os.path.join(t9.OOF, tag + ".npy"), "draws2": os.path.join(WORK, "oof_lgb2", tag + ".npy")}
    pl = np.load(lg["v9"]) if lgb_kind == "v9" else (
        np.load(lg["draws2"]) if lgb_kind == "draws2" else 0.5 * (np.load(lg["v9"]) + np.load(lg["draws2"])))
    g = [np.load(os.path.join(t9.GRU_OOF, tag + ".npy"))]
    if gru_kind == "seeds3":
        g += [np.load(os.path.join(WORK, "oof_gru_s{}".format(s), tag + ".npy")) for s in GRU_SEEDS]
    return pl, np.mean(g, axis=0)


BLEND = dict(w_lgb=[1.0, 0.75, 0.6, 0.5], c_team=[6.0, 8.0, 12.0], c_dead=[9.0, 12.0], min_dead=[1], scale=[1.0])
BLEND_BND = dict(w_lgb=[0.75, 0.6], c_team=[8.0, 12.0, 16.0, 24.0], c_dead=[9.0, 12.0], min_dead=[1], scale=[1.0],
                 bnd_a=[0.0, 0.5, 0.75, 0.9])


def _grid(cand):
    g = BLEND_BND if "bnd" in cand[2:] else BLEND
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


def oob_categories(cat, X, names):
    """Two more label-free evidence categories: ball seen > 1 m outside the lines (7), and a
    tracking gap whose last seen ball was outside (6)."""
    oob = X[:, names.index("g_oob")]
    vis = X[:, names.index("g_vis")] == 1
    last = np.where(vis, np.nan_to_num(oob), np.nan)
    idx = np.where(~np.isnan(last), np.arange(len(last)), 0)
    np.maximum.accumulate(idx, out=idx)
    prev = np.nan_to_num(last[idx])            # forward-filled 'last seen ball was out'
    out = cat.copy()
    out[(~vis) & (prev == 1)] = 6
    out[vis & (np.nan_to_num(oob) == 1)] = 7
    return out


def _blend_job(args):
    tag, cand, s = args
    pl, pg = _probs(tag, cand[0], cand[1])
    p = s["w_lgb"] * pl + (1 - s["w_lgb"]) * pg
    names, X, y, halves = _load(tag)
    graded = np.isin(y, ("H", "A", "D"))
    cat = t9._gap_cat_from_features(X, names)
    if "oob" in cand[2:]:
        cat = oob_categories(cat, X, names)
    ev, _ = model.em_evidence(cat, p[:, 2], graded)
    bd = np.load(os.path.join(WORK, "oof_bnd", tag + ".npy")) if s.get("bnd_a", 0) > 0 else None
    return model.spell_scores(y, model.decode_clip(p, halves, len(y), s, ev, bd))


def blend(jobs, cands):
    from multiprocessing import Pool
    tags = [a for a, _ in pairs()]
    work = [(t, c, s) for c in cands for s in _grid(c) for t in tags]
    with Pool(jobs) as pool:
        res = pool.map(_blend_job, work, chunksize=1)
    table, k = [], 0
    for c in cands:
        for s in _grid(c):
            per = {t: res[k + j] for j, t in enumerate(tags)}
            k += len(tags)
            table.append(dict(cand=c, setting=s, spell_f1=float(np.mean([v["spell_f1"] for v in per.values()])),
                              macro_f1=float(np.mean([v["macro_f1"] for v in per.values()])), per=per))
    report = {}
    for c in cands:
        rows = [r for r in table if r["cand"] == c]
        nested = []
        for t in tags:
            b = max(rows, key=lambda r: np.mean([v["spell_f1"] for u, v in r["per"].items() if u != t]))
            nested.append(b["per"][t]["spell_f1"])
        best = max(rows, key=lambda r: r["spell_f1"])
        report["/".join(c)] = dict(best=best, nested=float(np.mean(nested)),
                                   top=[dict(setting=r["setting"], spell_f1=r["spell_f1"]) for r in
                                        sorted(rows, key=lambda r: -r["spell_f1"])[:8]])
        print("{:20} best {:.4f} nested {:.4f} {}".format("/".join(c), best["spell_f1"], float(np.mean(nested)),
                                                          best["setting"]), flush=True)
        print("    per " + " ".join("{}:{:.3f}({}/{})".format(t[:9], best["per"][t]["spell_f1"], best["per"][t]["n_pred"],
                                                              best["per"][t]["n_gt"]) for t in tags), flush=True)
    name = "blend_" + "_".join("-".join(c) for c in cands) + ".json"
    json.dump(report, open(os.path.join(WORK, name), "w"), indent=1)


# ----------------------------------------------------------------------------- final models
ML = os.path.join(ROOT, "possession", "ml")


def final_bnd():
    """Boundary model on all 10 matches (draw 1, as validated) -> possession/ml/v10_bnd.txt."""
    import lightgbm as lgb
    Xs, ys = [], []
    for tag, _ in pairs():
        n, X, y, _h = _load(tag)
        a, b = _bnd_rows(n, X, y)
        Xs.append(a)
        ys.append(b)
    data = lgb.Dataset(np.vstack(Xs), label=np.concatenate(ys), feature_name=list(n), free_raw_data=True)
    del Xs
    booster = lgb.train(BND_PARAMS, data, num_boost_round=BND_ROUNDS)
    booster.save_model(os.path.join(ML, "v10_bnd.txt"), num_iteration=-1)
    print("final boundary model written")


def final_lgb2():
    """LightGBM on both degradation draws of all 10 matches -> possession/ml/v10_lgbm.txt."""
    import lightgbm as lgb
    Xs, ys = [], []
    for a_tag, b_tag in pairs():
        for tag in (a_tag, b_tag):
            n, X, y, _h = _load(tag)
            r = model.rows(X, y, n)
            Xs.append(r[0].astype(np.float32))
            ys.append(r[1])
            del X
    data = lgb.Dataset(np.vstack(Xs), label=np.concatenate(ys), feature_name=list(n), free_raw_data=True)
    del Xs
    booster = lgb.train(dict(model.LGB_PARAMS), data, num_boost_round=model.ROUNDS)
    booster.save_model(os.path.join(ML, "v10_lgbm.txt"), num_iteration=-1)
    print("final LightGBM (2 draws) written")


def final_gru(seed):
    """GRU on all 10 matches (draw 1) with `seed` (PyTorch environment) -> v10_gru_s{seed}.npz.
    Seed 0 with 8 threads is exactly v9's GRU."""
    from possession.ml import gru
    tags = [a for a, _ in pairs()]
    data = [_load(t) for t in tags]
    names = data[0][0]
    sc = gru.scaler([d[1] for d in data])
    seqs = [s for d in data for s in gru.sequences(d[1], d[2], d[3], sc)]
    t = time.time()
    w = gru.train(seqs, names, seed=seed, threads=8)
    path = os.path.join(ML, "v10_gru_s{}.npz".format(seed))
    np.savez(path, scale_median=sc[0], scale_iqr=sc[1], **{k.replace(".", "__"): v for k, v in w.items()})
    print("final GRU seed", seed, "written in {:.0f}s".format(time.time() - t))


def write_config(cand, setting_json):
    """possession/ml/v10_config.json from the chosen candidate and blend setting."""
    from possession.ml import sources
    s = json.load(open(setting_json))["/".join(cand)]
    best = dict(s["best"]["setting"])
    w_lgb = best.pop("w_lgb")
    lgb_files = ["v10_lgbm.txt"] if cand[0] == "draws2" else ["v9_lgbm.txt"]
    gru_files = ["v10_gru_s{}.npz".format(k) for k in ((0,) + GRU_SEEDS if cand[1] == "seeds3" else (0,))]
    cfg = dict(version="v10", lgb_models=lgb_files, gru_models=gru_files,
               bnd_model="v10_bnd.txt" if "bnd" in cand[2:] else None,
               blend_w_lgb=w_lgb, decoder=best, use_gap_evidence=True, use_cut_evidence=True,
               evidence_weight=model.EVIDENCE_WEIGHT,
               oof_spell_f1=s["best"]["spell_f1"], oof_macro_f1=s["best"]["macro_f1"], nested_spell_f1=s["nested"],
               per_match={t: dict(spell_f1=v["spell_f1"], n_pred=v["n_pred"], n_gt=v["n_gt"])
                          for t, v in s["best"]["per"].items()},
               trained_on=[a for a, _ in pairs()] + ([b for _, b in pairs()] if cand[0] == "draws2" else []),
               lgb_params=model.LGB_PARAMS, rounds=model.ROUNDS, bnd_params=BND_PARAMS, bnd_rounds=BND_ROUNDS,
               bnd_window=BND_WIN, gru_seeds=[0] + (list(GRU_SEEDS) if cand[1] == "seeds3" else []))
    files = lgb_files + gru_files + ([cfg["bnd_model"]] if cfg["bnd_model"] else [])
    cfg["sha256"] = {f: sources.sha256(os.path.join(ML, f)) for f in files}
    cfg["feature_names"] = _load(pairs()[0][0])[0]
    json.dump(cfg, open(os.path.join(ML, "v10_config.json"), "w", newline=chr(10)), indent=1)
    print("v10_config.json written:", {k: cfg[k] for k in ("lgb_models", "gru_models", "bnd_model", "blend_w_lgb", "decoder")})



def skillcorner_check():
    """Untouched check: the final v10 configuration on the 20 SkillCorner matches."""
    from possession.ml import features, gru, sources
    cfg = json.load(open(os.path.join(ML, "v10_config.json")))
    boosters = [model.load_booster(os.path.join(ML, f)) for f in cfg["lgb_models"]]
    grus = [gru.load(os.path.join(ML, f)) for f in cfg["gru_models"]]
    bnd = model.load_booster(os.path.join(ML, cfg["bnd_model"]))
    st = t9.stats()
    rows = []
    for mid in sources.skillcorner_ids():
        tag, names, X, y, halves = model.prepared("skillcorner", mid, st, model.clip_seed("skillcorner", mid))
        p = cfg["blend_w_lgb"] * np.mean([model.predict(b, X, names=names) for b in boosters], axis=0)
        p = p + (1 - cfg["blend_w_lgb"]) * np.mean([gru.predict(w, sc, X, halves, names) for w, sc in grus], axis=0)
        bd = bnd_predict(bnd, X, names)
        graded = np.zeros(len(y), bool)
        for _, a, b in halves:
            graded[a:b + 1] = True
        ev, _ = model.em_evidence(t9._gap_cat_from_features(X, names), p[:, 2], graded)
        sc_ = model.spell_scores(y, model.decode_clip(p, halves, len(y), cfg["decoder"], ev, bd))
        rows.append(sc_)
        print("  SC {} spellF1 {:.4f} poss {}/{}".format(mid, sc_["spell_f1"], sc_["n_pred"], sc_["n_gt"]), flush=True)
    print("SkillCorner mean spellF1 {:.4f} over {} matches".format(np.mean([r["spell_f1"] for r in rows]), len(rows)))
    json.dump(rows, open(os.path.join(WORK, "skillcorner_v10.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("prep2", "lgb2", "gru_seeds", "bnd_cv", "blend", "final_bnd", "final_lgb2",
                                      "final_gru", "config", "skillcorner"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cand", default="v9/seed0/bnd")
    ap.add_argument("--blend-json", default=None)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--cands", default="v9/seed0,v9/seeds3")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    if a.stage == "prep2":
        prep2()
    elif a.stage == "lgb2":
        lgb2()
    elif a.stage == "gru_seeds":
        gru_seeds(a.jobs)
    elif a.stage == "bnd_cv":
        bnd_cv()
    elif a.stage == "final_bnd":
        final_bnd()
    elif a.stage == "final_lgb2":
        final_lgb2()
    elif a.stage == "final_gru":
        final_gru(a.seed)
    elif a.stage == "skillcorner":
        skillcorner_check()
    elif a.stage == "config":
        write_config(tuple(a.cand.split("/")), a.blend_json)
    else:
        blend(a.jobs, [tuple(c.split("/")) for c in a.cands.split(",")])


if __name__ == "__main__":
    main()
