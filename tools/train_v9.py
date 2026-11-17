"""Train and select v9 on public external matches only, then run it on BAR-ATH.

    python tools/train_v9.py prepare     # degrade + features for every external match (cached)
    python tools/train_v9.py cv          # leave-one-match-out LightGBM, out-of-fold probabilities
    python tools/train_v9.py grid        # decoder settings chosen on the out-of-fold matches
    python tools/train_v9.py final       # train on every match, predict BAR-ATH, write the model
    python tools/train_v9.py skillcorner # untouched check on 20 SkillCorner matches (H/A only)

Protocol (fixed before any BAR-ATH run; RULES.md rule 4):
- training/validation: 7 DFL/IDSSE matches (degraded to BAR-ATH's tracking statistics) and the 3
  Metrica sample games; labels aligned to controlled possession (possession/ml/labels.py);
- every model and decoder choice is made on leave-one-match-out spellF1, averaged over the held-out
  matches; the BAR-ATH public window is scored once at the end, for reporting only;
- SkillCorner is never trained on and never used to choose anything.
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
from possession.ml import degrade, model, sources  # noqa: E402

WORK = os.path.join(sources.CACHE, "v9")
OOF = os.path.join(WORK, "oof")
GRID = dict(c_team=[2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0], c_dead=[2.0, 4.0, 6.0, 9.0, 12.0],
            min_dead=[1, 25, 50], scale=[1.0])
SEED = 0


def stats():
    path = os.path.join(sources.CACHE, "barath_tracking_stats_bc.npz")
    if os.path.isfile(path):
        z = np.load(path)
        return z["vis"], z["hc"], z["ac"]
    # the ball mask the model will meet on BAR-ATH: tactical ball filled from the broadcast
    vis, hc, ac = degrade.barath_stats(sources.barath_clip(use_broadcast=True))
    np.savez(path, vis=vis, hc=hc, ac=ac)
    return vis, hc, ac


def prepare_all():
    st = stats()
    out = []
    for kind, key in model.training_clips():
        t = time.time()
        tag, names, X, y, halves = model.prepared(kind, key, st, model.clip_seed(kind, key))
        out.append((tag, names, X, y, halves))
        print("prepared", tag, X.shape, "{:.0f}s".format(time.time() - t), flush=True)
    return out


def cv():
    data = prepare_all()
    os.makedirs(OOF, exist_ok=True)
    for i, (tag, names, X, y, halves) in enumerate(data):
        path = os.path.join(OOF, tag + ".npy")
        if os.path.isfile(path):
            continue
        t = time.time()
        booster = model.train([(d[1], d[2], d[3]) for j, d in enumerate(data) if j != i])
        np.save(path, model.predict(booster, X, names=names))
        print("fold", tag, "{:.0f}s".format(time.time() - t), flush=True)


def _decode_job(args):
    tag, setting = args
    p = np.load(os.path.join(OOF, tag + ".npy"))
    z = np.load(os.path.join(model.FEATURE_CACHE, tag + ".npz"), allow_pickle=False)
    y, halves = z["y"], [tuple(h) for h in z["halves"]]
    pred = model.decode_clip(p, halves, len(y), setting)
    return model.spell_scores(y, pred)


def grid(jobs):
    from multiprocessing import Pool
    tags = sorted(f[:-4] for f in os.listdir(OOF) if f.endswith(".npy"))
    settings = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    work = [(t, s) for s in settings for t in tags]
    with Pool(jobs) as pool:
        res = pool.map(_decode_job, work, chunksize=1)
    table = []
    for k, s in enumerate(settings):
        per = {t: res[k * len(tags) + j] for j, t in enumerate(tags)}
        table.append(dict(setting=s, spell_f1=float(np.mean([v["spell_f1"] for v in per.values()])),
                          macro_f1=float(np.mean([v["macro_f1"] for v in per.values()])), per=per))
    # nested estimate: choose on all-but-one match, score the one left out
    nested = []
    for t in tags:
        best = max(table, key=lambda r: (np.mean([v["spell_f1"] for u, v in r["per"].items() if u != t]),))
        nested.append(best["per"][t]["spell_f1"])
    table.sort(key=lambda r: -r["spell_f1"])
    report = dict(tags=tags, best=table[0], nested_spell_f1=float(np.mean(nested)),
                  top=[dict(setting=r["setting"], spell_f1=r["spell_f1"], macro_f1=r["macro_f1"]) for r in table[:15]])
    os.makedirs(WORK, exist_ok=True)
    json.dump(report, open(os.path.join(WORK, "grid.json"), "w"), indent=1)
    print("best", table[0]["setting"], "spellF1 {:.4f} macroF1 {:.4f}".format(table[0]["spell_f1"], table[0]["macro_f1"]))
    print("nested (honest) spellF1 {:.4f}".format(report["nested_spell_f1"]))
    for t in tags:
        v = table[0]["per"][t]
        print("  {:28} spellF1 {:.4f}  macroF1 {:.4f}  poss {}/{}".format(t, v["spell_f1"], v["macro_f1"], v["n_pred"], v["n_gt"]))


def _gap_cat_from_features(X, names):
    vis = X[:, names.index("g_vis")]
    gap = X[:, names.index("g_gap")]
    cat = np.where(vis == 1, 0, 1 + np.searchsorted(model.GAP_BINS, np.nan_to_num(gap), side="right") - 1)
    return cat.astype(np.int8)


def _evidence_job(args):
    tag, setting, use = args
    p = np.load(os.path.join(OOF, tag + ".npy"))
    z = np.load(os.path.join(model.FEATURE_CACHE, tag + ".npz"), allow_pickle=False)
    y, halves, names, X = z["y"], [tuple(h) for h in z["halves"]], list(z["names"]), z["X"]
    graded = np.isin(y, ("H", "A", "D"))
    ev = None
    if use:
        ev, _ = model.em_evidence(_gap_cat_from_features(X, names), p[:, 2], graded)
    pred = model.decode_clip(p, halves, len(y), setting, ev)
    return tag, use, model.spell_scores(y, pred)


def evidence(jobs):
    """Does the label-free gap evidence help on held-out matches? (Metrica keeps its own ball gaps,
    so it is the real test; DFL's gaps are copied independently of its labels, a null check.)"""
    from multiprocessing import Pool
    best = json.load(open(os.path.join(WORK, "grid.json")))["best"]["setting"]
    tags = sorted(f[:-4] for f in os.listdir(OOF) if f.endswith(".npy"))
    with Pool(jobs) as pool:
        res = pool.map(_evidence_job, [(t, best, u) for t in tags for u in (False, True)])
    out = {}
    for tag, use, sc in res:
        out.setdefault(tag, {})["with" if use else "without"] = sc
    for t, v in out.items():
        print("  {:24} without {:.4f}  with {:.4f}".format(t, v["without"]["spell_f1"], v["with"]["spell_f1"]))
    for src in ("metrica", "dfl"):
        a = [v["without"]["spell_f1"] for t, v in out.items() if t.startswith(src)]
        b = [v["with"]["spell_f1"] for t, v in out.items() if t.startswith(src)]
        if a:
            print("{:8} mean spellF1 without {:.4f}  with {:.4f}".format(src, np.mean(a), np.mean(b)))
    json.dump(out, open(os.path.join(WORK, "evidence.json"), "w"), indent=1)


def final(use_gap, use_cut):
    """Train on every external match; write the model and its config (no BAR-ATH labels exist here)."""
    data = prepare_all()
    t = time.time()
    booster = model.train([(d[1], d[2], d[3]) for d in data])
    path = os.path.join(ROOT, "possession", "ml", "v9_lgbm.txt")
    model.save_booster(booster, path)
    g = json.load(open(os.path.join(WORK, "blend.json")))
    best = dict(g["best"]["setting"])
    w_lgb = best.pop("w_lgb")
    cfg = dict(decoder=best, blend_w_lgb=w_lgb, oof_spell_f1=g["best"]["spell_f1"],
               oof_macro_f1=g["best"]["macro_f1"], nested_spell_f1=g["nested_spell_f1"],
               use_gap_evidence=use_gap,
               use_cut_evidence=use_cut, evidence_weight=model.EVIDENCE_WEIGHT,
               trained_on=[d[0] for d in data], lgb_params=model.LGB_PARAMS, rounds=model.ROUNDS,
               model_sha256=sources.sha256(path), feature_names=data[0][1])
    json.dump(cfg, open(os.path.join(ROOT, "possession", "ml", "v9_config.json"), "w", newline=chr(10)), indent=1)
    print("final model trained on", len(data), "matches in {:.0f}s".format(time.time() - t))


def final_gru():
    """Train the GRU on every external match (PyTorch environment); write numpy weights."""
    from possession.ml import gru
    data = prepare_all()
    names = data[0][1]
    sc = gru.scaler([d[2] for d in data])
    seqs = [s for d in data for s in gru.sequences(d[2], d[3], d[4], sc)]
    t = time.time()
    w = gru.train(seqs, names, seed=0, threads=8)
    path = os.path.join(ROOT, "possession", "ml", "v9_gru.npz")
    np.savez(path, scale_median=sc[0], scale_iqr=sc[1], **{k.replace(".", "__"): v for k, v in w.items()})
    cfg_path = os.path.join(ROOT, "possession", "ml", "v9_config.json")
    cfg = json.load(open(cfg_path))
    import torch
    cfg.update(gru_sha256=sources.sha256(path), gru_epochs=gru.EPOCHS, gru_hidden=gru.HIDDEN,
               gru_trained_with="torch " + torch.__version__)
    json.dump(cfg, open(cfg_path, "w", newline=chr(10)), indent=1)
    print("final GRU trained in {:.0f}s".format(time.time() - t))


def skillcorner(jobs):
    """Untouched check: the final model on SkillCorner (never trained on, never used to choose)."""
    cfg = json.load(open(os.path.join(ROOT, "possession", "ml", "v9_config.json")))
    from possession.ml import gru
    booster = model.load_booster(os.path.join(ROOT, "possession", "ml", "v9_lgbm.txt"))
    w, scl = gru.load(os.path.join(ROOT, "possession", "ml", "v9_gru.npz"))
    st = stats()
    rows = []
    for mid in sources.skillcorner_ids():
        tag, names, X, y, halves = model.prepared("skillcorner", mid, st, model.clip_seed("skillcorner", mid))
        p = cfg["blend_w_lgb"] * model.predict(booster, X, names=names)
        p = p + (1 - cfg["blend_w_lgb"]) * gru.predict(w, scl, X, halves, names)
        graded = np.zeros(len(y), bool)
        for _, a, b in halves:
            graded[a:b + 1] = True
        ev, _ = model.em_evidence(_gap_cat_from_features(X, names), p[:, 2], graded) if cfg["use_gap_evidence"] else (None, None)
        pred = model.decode_clip(p, halves, len(y), cfg["decoder"], ev)
        sc = model.spell_scores(y, pred)
        rows.append(sc)
        print("  SC {} spellF1 {:.4f} macroF1(H/A only) {:.4f} poss {}/{}".format(mid, sc["spell_f1"], sc["macro_f1"], sc["n_pred"], sc["n_gt"]), flush=True)
    print("SkillCorner mean spellF1 {:.4f} over {} matches".format(np.mean([r["spell_f1"] for r in rows]), len(rows)))
    json.dump(rows, open(os.path.join(WORK, "skillcorner.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("prepare", "cv", "grid", "evidence", "gru_cv", "blend", "final", "final_gru", "skillcorner"))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--gap", type=int, default=1, help="final: use the gap evidence (1/0)")
    ap.add_argument("--cut", type=int, default=1, help="final: use the broadcast cut-away evidence (1/0)")
    a = ap.parse_args()
    if a.stage == "prepare":
        prepare_all()
    elif a.stage == "cv":
        cv()
    elif a.stage == "grid":
        grid(a.jobs)
    elif a.stage == "evidence":
        evidence(a.jobs)
    elif a.stage == "gru_cv":
        gru_cv(a.jobs)
    elif a.stage == "blend":
        blend(a.jobs)
    elif a.stage == "final":
        final(bool(a.gap), bool(a.cut))
    elif a.stage == "final_gru":
        final_gru()
    else:
        skillcorner(a.jobs)



# ----------------------------------------------------------------------------- GRU stage
GRU_OOF = os.path.join(WORK, "oof_gru")


def _gru_fold(args):
    i, tags = args
    from possession.ml import gru
    held = tags[i]
    data = {}
    for t in tags:
        z = np.load(os.path.join(model.FEATURE_CACHE, t + ".npz"), allow_pickle=False)
        data[t] = (list(z["names"]), z["X"], z["y"], [tuple(h) for h in z["halves"]])
    names = data[held][0]
    train_tags = [t for t in tags if t != held]
    sc = gru.scaler([data[t][1] for t in train_tags])
    seqs = [s for t in train_tags for s in gru.sequences(data[t][1], data[t][2], data[t][3], sc)]
    w = gru.train(seqs, names, seed=0, threads=3)
    p = gru.predict(w, sc, data[held][1], data[held][3], names)
    np.save(os.path.join(GRU_OOF, held + ".npy"), p)
    return held


def gru_cv(jobs):
    from multiprocessing import Pool
    os.makedirs(GRU_OOF, exist_ok=True)
    tags = sorted(f[:-4] for f in os.listdir(OOF) if f.endswith(".npy"))
    todo = [i for i, t in enumerate(tags) if not os.path.isfile(os.path.join(GRU_OOF, t + ".npy"))]
    t0 = time.time()
    with Pool(jobs) as pool:
        for held in pool.imap_unordered(_gru_fold, [(i, tags) for i in todo]):
            print("gru fold", held, "{:.0f}s".format(time.time() - t0), flush=True)


BLEND = dict(w_lgb=[1.0, 0.75, 0.5, 0.25, 0.0], c_team=[6.0, 8.0, 12.0, 16.0, 24.0],
             c_dead=[6.0, 9.0, 12.0], min_dead=[1, 25], scale=[1.0])


def _blend_job(args):
    tag, s, use_gap = args
    p = s["w_lgb"] * np.load(os.path.join(OOF, tag + ".npy"))
    if s["w_lgb"] < 1:
        p = p + (1 - s["w_lgb"]) * np.load(os.path.join(GRU_OOF, tag + ".npy"))
    z = np.load(os.path.join(model.FEATURE_CACHE, tag + ".npz"), allow_pickle=False)
    y, halves = z["y"], [tuple(h) for h in z["halves"]]
    ev = None
    if use_gap:
        graded = np.isin(y, ("H", "A", "D"))
        ev, _ = model.em_evidence(_gap_cat_from_features(z["X"], list(z["names"])), p[:, 2], graded)
    return model.spell_scores(y, model.decode_clip(p, halves, len(y), s, ev))


def blend(jobs):
    """Choose the LightGBM/GRU blend weight and the decoder together, on held-out matches, with
    the gap evidence on (it was adopted on its own test)."""
    from multiprocessing import Pool
    tags = sorted(f[:-4] for f in os.listdir(GRU_OOF) if f.endswith(".npy"))
    settings = [dict(zip(BLEND, v)) for v in itertools.product(*BLEND.values())]
    work = [(t, s, True) for s in settings for t in tags]
    with Pool(jobs) as pool:
        res = pool.map(_blend_job, work, chunksize=1)
    table = []
    for k, s in enumerate(settings):
        per = {t: res[k * len(tags) + j] for j, t in enumerate(tags)}
        table.append(dict(setting=s, spell_f1=float(np.mean([v["spell_f1"] for v in per.values()])),
                          macro_f1=float(np.mean([v["macro_f1"] for v in per.values()])), per=per))
    nested = []
    for t in tags:
        best = max(table, key=lambda r: np.mean([v["spell_f1"] for u, v in r["per"].items() if u != t]))
        nested.append(best["per"][t]["spell_f1"])
    by_w = {}
    for r in table:
        w = r["setting"]["w_lgb"]
        if w not in by_w or r["spell_f1"] > by_w[w]["spell_f1"]:
            by_w[w] = r
    table.sort(key=lambda r: -r["spell_f1"])
    report = dict(tags=tags, best=table[0], nested_spell_f1=float(np.mean(nested)),
                  best_by_weight={str(w): dict(setting=r["setting"], spell_f1=r["spell_f1"], macro_f1=r["macro_f1"])
                                  for w, r in sorted(by_w.items())})
    json.dump(report, open(os.path.join(WORK, "blend.json"), "w"), indent=1)
    for w, r in sorted(by_w.items()):
        print("w_lgb {:.2f}: best spellF1 {:.4f} macroF1 {:.4f} {}".format(w, r["spell_f1"], r["macro_f1"], r["setting"]))
    print("best", table[0]["setting"], "spellF1 {:.4f}; nested {:.4f}".format(table[0]["spell_f1"], report["nested_spell_f1"]))
    for t in tags:
        v = table[0]["per"][t]
        print("  {:24} spellF1 {:.4f} macroF1 {:.4f} poss {}/{}".format(t, v["spell_f1"], v["macro_f1"], v["n_pred"], v["n_gt"]))


if __name__ == "__main__":
    main()
