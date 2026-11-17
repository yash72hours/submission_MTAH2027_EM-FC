"""Download the public external training/validation data into data/external/ (git-ignored).

    python tools/download_external.py            # DFL/IDSSE (~2.6 GB) + Metrica 1-3 (~0.2 GB)
    python tools/download_external.py --skillcorner   # also the 20 SkillCorner check matches (~1.9 GB)

Only needed to retrain or re-validate the models; `make_submission.py` uses the committed models.
Every file is checked against the sizes pinned in possession/ml/external_manifest.json (DFL) or the
SHA-256 values in possession/external.py (Metrica 1-2).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import external  # noqa: E402
from possession.ml import sources  # noqa: E402

MANIFEST = json.load(open(os.path.join(ROOT, "possession", "ml", "external_manifest.json")))
IDSSE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{}/".format(MANIFEST["idsse"]["revision"])
METRICA = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/data/Sample_Game_3/".format(
    MANIFEST["metrica"]["commit"])
SC_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/"
SC_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/"
SC_IDS = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436,
          2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461)


def fetch(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    if shutil.which("curl"):
        subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", tmp, url], check=True)
    else:
        urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    print("  got", os.path.relpath(path, ROOT), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skillcorner", action="store_true")
    a = ap.parse_args()
    print("DFL / IDSSE (CC BY 4.0)")
    for name, info in MANIFEST["idsse"]["files"].items():
        path = os.path.join(sources.DFL_DIR, name)
        fetch(IDSSE + name, path)
        if os.path.getsize(path) != info["size"]:
            raise SystemExit("size mismatch: {}".format(name))
    print("Metrica sample games 1-2")
    external.download()
    print("Metrica sample game 3")
    d = os.path.join(sources.EXTERNAL, "metrica_" + MANIFEST["metrica"]["commit"], "Sample_Game_3")
    for f in ("Sample_Game_3_tracking.txt", "Sample_Game_3_events.json", "Sample_Game_3_metadata.xml"):
        fetch(METRICA + f, os.path.join(d, f))
    if a.skillcorner:
        print("SkillCorner open data (MIT; evaluation only)")
        d = os.path.join(sources.EXTERNAL, "skillcorner")
        fetch(SC_RAW + "matches.json", os.path.join(d, "matches.json"))
        for i in SC_IDS:
            fetch("{}matches/{}/{}_match.json".format(SC_RAW, i, i), os.path.join(d, str(i), "{}_match.json".format(i)))
            fetch("{}matches/{}/{}_tracking_extrapolated.jsonl".format(SC_LFS, i, i),
                  os.path.join(d, str(i), "{}_tracking_extrapolated.jsonl".format(i)))
    print("done")


if __name__ == "__main__":
    main()
