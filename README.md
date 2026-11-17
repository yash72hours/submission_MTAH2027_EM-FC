# EM FC: ball possession from automatic tracking (More than a Hack 2027)

This repository labels every frame of FC Barcelona vs Athletic Club as `H`, `A`, `D` or `X`.
It uses a frame model trained only on public tracking data from **other matches**, adapted to this
match's tracking quality, and decoded into possession spells.

- **Technical document:** [EM_FC_technical_document.pdf](EM_FC_technical_document.pdf) (5 pages, as submitted); a longer version is in [TECHNICAL.md](TECHNICAL.md)
- **Submitted file:** `EM FC.csv`, SHA-256 `9b029648d599dc6e310fdb2ba59c1f40ea6ac3631a5d84536a304ccaf1aba23d`

## Reproduce the submission

**Requirements:** Python 3.11 or 3.12, CPU only, about 1 minute.

```bash
pip install -r requirements.txt
# Linux only: LightGBM needs OpenMP:  sudo apt install libgomp1
```

**Data.** Put the challenge data bundle (`tactical/` and `broadcast/` folders) in `./data`, or
point `MTAH_DATA_DIR` at it. Only the tracking and homography files are read; no mp4 is needed.

```bash
python make_submission.py                                  # -> evaluation/results/EM FC.csv
python evaluation/validate_submission.py "evaluation/results/EM FC.csv"
python evaluation/run_evaluation.py --bootstrap 1000      # scores evaluation/results/*.csv on the public 5 minutes (a check only)
```

The output is byte-identical on Windows (Python 3.11) and Linux (Python 3.12), starting from empty
caches.

## What is in here

| path | what |
|---|---|
| `make_submission.py` | the one command that regenerates the file |
| `possession/ml/infer.py` | inference: tracking → features → models → match evidence → decoder |
| `possession/ml/sources.py` | loaders: the provided game, DFL/IDSSE, Metrica, SkillCorner |
| `possession/ml/broadcast.py` | broadcast-to-tactical synchronisation, ball gap-fill, cut-away flag |
| `possession/ml/degrade.py` | makes clean external tracking look like the provided game's |
| `possession/ml/features.py` | 219 team-symmetric features with temporal context |
| `possession/ml/labels.py` | external labels → controlled possession (2-s touch rule) |
| `possession/ml/model.py`, `gru.py` | LightGBM, bidirectional GRU (numpy inference), decoder glue, EM evidence |
| `possession/decode.py` | Viterbi decoders |
| `possession/data.py`, `games.py` | provided-data loading (organisers' layout) |
| `possession/external.py` | Metrica sample-data loader and event-derived labels |
| `possession/ml/*.txt, *.npz, *_config.json` | the trained models and their configuration |
| `possession/ml/external_manifest.json` | pinned external datasets (revision, sizes) |
| `tools/download_external.py` | downloads the external data into `data/external/` |
| `tools/train_v9.py`, `tools/exp_v10.py` | training, leave-one-match-out validation, model selection |
| `evaluation/` | the organisers' validator and scorer (MIT, unchanged) |

## Retrain the models (optional)

The models are committed, so retraining is not needed to reproduce the file.

```bash
pip install -r requirements-train.txt          # adds PyTorch (CPU) for the GRU
python tools/download_external.py              # ~2.8 GB of public external data (add --skillcorner for the check set)

# 1. degraded training copies and features (needs the provided tracking for its statistics)
python tools/train_v9.py prepare               # degradation draw 1 of the 10 external matches
python tools/exp_v10.py prep2                  # degradation draw 2

# 2. leave-one-match-out validation (held-out probabilities for every model)
python tools/train_v9.py cv                    # LightGBM, draw 1
python tools/exp_v10.py lgb2                   # LightGBM, both draws
python tools/train_v9.py gru_cv                # GRU seed 0      (PyTorch environment)
python tools/exp_v10.py gru_seeds              # GRU seeds 1, 2  (PyTorch environment)
python tools/exp_v10.py bnd_cv                 # possession-boundary model

# 3. choose blend weight and decoder costs jointly on the held-out matches
python tools/exp_v10.py blend --cands "draws2/seeds3/bnd"

# 4. final models on all 10 matches, and the configuration
python tools/exp_v10.py final_lgb2
python tools/exp_v10.py final_bnd
python tools/exp_v10.py final_gru --seed 0     # and --seed 1, --seed 2 (PyTorch environment)
python tools/exp_v10.py config --cand draws2/seeds3/bnd --blend-json data/external/cache/v9/v10/blend_draws2-seeds3-bnd.json
```

LightGBM runs in deterministic mode, and the GRU training is seeded with deterministic PyTorch
kernels, so retraining reproduces the committed model files.

## Rules compliance

- **Training and tuning data.** Every model and every parameter was trained or tuned on public
  external matches only: DFL/IDSSE and the Metrica sample games. SkillCorner was used only as an
  untouched check.
- **The provided game.** Its ground truth and its video were never used for training, fitting,
  calibration or selection. No hand labels or other human-produced data about it were used.
- **The public 5 minutes** were scored once, at the end, as a pipeline check.
- **No data in this repository.** Neither match data nor external data is included; see
  `.gitignore`.

## Licences and credits

- **Code:** MIT (see `LICENSE`).
- **`evaluation/`, `games.py`:** from the organisers' challenge repository (MIT).
- **DFL / IDSSE open data:** Bassek, Rein, Weber & Memmert (2025), *Scientific Data*, CC BY 4.0.
- **Metrica Sports sample data:** https://github.com/metrica-sports/sample-data.
- **SkillCorner open data:** MIT, https://github.com/SkillCorner/opendata.
