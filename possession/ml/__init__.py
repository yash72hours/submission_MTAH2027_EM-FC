"""The EM FC model: a frame model learned on public external matches, decoded with Viterbi.

    sources.py    one Clip format for BAR-ATH and the external matches (DFL/IDSSE, Metrica, SkillCorner)
    labels.py     external labels aligned to controlled possession (touches are not possessions)
    degrade.py    makes clean external tracking look like BAR-ATH's automatic tracking
    broadcast.py  broadcast-to-tactical sync, ball gap-fill, cut-away flag
    features.py   team-level, permutation-invariant features with multi-scale context
    model.py      LightGBM frame and boundary models, decoder glue, label-free EM evidence
    gru.py        bidirectional GRU (PyTorch training, numpy inference)
    infer.py      the submitted pipeline on BAR-ATH (configuration: v10_config.json)

Nothing here is trained, fitted or calibrated on the BAR-ATH ground truth or video. BAR-ATH enters
only as input tracking at inference, and its tracking statistics (not labels) set the degrader.
"""
