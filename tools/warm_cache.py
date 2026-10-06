"""Pre-compute (cache) the analysis of dataset clips, e.g. after adding new
ones. Usage: python tools/warm_cache.py [--workers 2] [--match PATTERN]"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS, LIB = ROOT / "data/dataset", ROOT / "data/library"


def run(clip):
    from speechlens.pipeline import analyze_cached
    text = clip.get("transcript") or (LIB / clip["source"] / "transcript.txt").read_text()
    analyze_cached(audio.load(audio.clip_path(DS / "clips", clip["id"])), text)
    return clip["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--match", default="")
    a = ap.parse_args()
    clips = [c for c in json.loads((DS / "manifest.json").read_text())["clips"] if a.match in c["id"]]
    os.environ.setdefault("SPEECHLENS_THREADS", str(max(1, (os.cpu_count() or 1) // a.workers)))
    with ProcessPoolExecutor(a.workers) as ex:
        for i, cid in enumerate(ex.map(run, clips)):
            print(f"{i + 1}/{len(clips)} {cid}", flush=True)


if __name__ == "__main__":
    main()
