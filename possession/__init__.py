"""Frame-level ball possession for the More than a Hack 2027 challenge (team EM FC).

    ml/          the model: loaders, degradation, features, LightGBM + GRU, boundary model, inference
    decode.py    Viterbi decoders (fixed and per-frame team-switch cost)
    data.py      the provided game's tracking, metadata and homography
    external.py  Metrica sample-data loader and event-derived possession labels
"""
