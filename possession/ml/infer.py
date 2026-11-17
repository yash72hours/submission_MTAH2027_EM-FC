"""The submitted model on BAR-ATH: features, the committed models, label-free match evidence, the
decoder.

Reads the BAR-ATH tactical tracking, its metadata and the broadcast tracking and homography. It
never reads the video, the Smart Tagging or any ground truth.

The configuration (`v10_config.json`, or `v9_config.json` for the earlier version) names the
models and the decoder settings, all chosen on held-out external matches (tools/exp_v10.py).
"""

import json
import os

import numpy as np

from . import features, model, sources

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "v10_config.json")
# FAQ.md: the second half is graded from 110,808 but kicks off at 113,742; those frames are dead.
KNOWN_DEAD = (110808, 113742)


def _check(path, expected):
    if sources.sha256(path) != expected:
        raise ValueError("{} differs from the file the configuration describes".format(path))


def _files(cfg):
    """(lightgbm files, gru files, boundary file or None) with their expected SHA-256."""
    if "lgb_models" in cfg:
        sha = cfg["sha256"]
        lg = [(os.path.join(HERE, f), sha[f]) for f in cfg["lgb_models"]]
        gr = [(os.path.join(HERE, f), sha[f]) for f in cfg["gru_models"]]
        bd = (os.path.join(HERE, cfg["bnd_model"]), sha[cfg["bnd_model"]]) if cfg.get("bnd_model") else None
        return lg, gr, bd
    # v9 layout
    return ([(os.path.join(HERE, "v9_lgbm.txt"), cfg["model_sha256"])],
            [(os.path.join(HERE, "v9_gru.npz"), cfg["gru_sha256"])], None)


def run(return_details=False, config=CONFIG):
    cfg = json.load(open(config))
    lg_files, gru_files, bnd_file = _files(cfg)
    for path, sha in lg_files + gru_files + ([bnd_file] if bnd_file else []):
        _check(path, sha)
    clip = sources.barath_clip(use_broadcast=True)
    names, X = features.compute(clip)
    if names != cfg["feature_names"]:
        raise ValueError("feature list differs from the one the models were trained with")

    from . import gru
    p_lgb = np.mean([model.predict(model.load_booster(path), X, names=names) for path, _ in lg_files], axis=0)
    p_gru = np.mean([gru.predict(*gru.load(path), X, clip.halves, names) for path, _ in gru_files], axis=0)
    w = cfg["blend_w_lgb"]
    p = w * p_lgb + (1 - w) * p_gru

    boundary = None
    if bnd_file:
        booster = model.load_booster(bnd_file[0])
        q = booster.predict(X, num_threads=4)
        q2 = booster.predict(X[:, features.swap_index(names)], num_threads=4)
        boundary = (0.5 * (q + q2)).astype(np.float32)

    graded = np.zeros(clip.n, bool)
    for _, a, b in clip.halves:
        graded[a:b + 1] = True
    evidence = np.zeros(clip.n, np.float32)
    tables = {}
    if cfg["use_gap_evidence"]:
        ev, tables["gap"] = model.em_evidence(model.gap_category(clip.ball), p[:, 2], graded,
                                              cfg["evidence_weight"])
        evidence += ev
    if cfg["use_cut_evidence"]:
        cut = np.nan_to_num(clip.extra["cut"], nan=2).astype(np.int8)
        ev, tables["cut"] = model.em_evidence(cut, p[:, 2], graded, cfg["evidence_weight"])
        evidence += ev
    labels = model.decode_clip(p, clip.halves, clip.n, cfg["decoder"], evidence, boundary)
    labels[KNOWN_DEAD[0]:KNOWN_DEAD[1]] = "D"
    if return_details:
        return labels, dict(halves=clip.halves, probabilities=p, boundary=boundary, evidence_tables=tables,
                            lags=clip.extra.get("lags"), filled=clip.extra.get("filled"))
    return labels, dict(halves=clip.halves)
