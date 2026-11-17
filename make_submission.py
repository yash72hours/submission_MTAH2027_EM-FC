"""Regenerate the EM FC submission from the provided data.

    python make_submission.py                   # -> evaluation/results/EM FC.csv
    python make_submission.py --name try --out somewhere/

Reads the BAR-ATH tactical tracking and the broadcast tracking + homography (no video, no ground
truth), runs the committed models in possession/ml/ and decodes possession spells.
Deterministic: the same data give a byte-identical file on every platform we tested.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DEFAULT_NAME = "EM FC"
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def summarise(labels, halves, fps=25):
    """What the file claims: possessions, durations, dead-ball share, home share."""
    spells, dead, graded = [], 0, 0
    for _, first, last in halves:
        window = labels[first:last + 1]
        graded += len(window)
        dead += int(np.sum(window == "D"))
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if window[a] in ("H", "A"):
                spells.append((window[a], b - a))
    lengths = np.array([n for _, n in spells]) / fps
    home = sum(n for team, n in spells if team == "H")
    alive = sum(n for _, n in spells)
    print("  possessions {:>4}  (H {}, A {})   mean {:.1f} s   median {:.1f} s".format(
        len(spells), sum(t == "H" for t, _ in spells), sum(t == "A" for t, _ in spells),
        lengths.mean(), np.median(lengths)))
    print("  dead ball {:.1f}% of graded frames   home share of alive {:.1f}%".format(
        100.0 * dead / graded, 100.0 * home / max(alive, 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()

    from possession.ml import infer
    labels, extras = infer.run()
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    print("wrote", path)
    print("  sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
    summarise(labels, extras["halves"])
