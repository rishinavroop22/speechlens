"""Add the reading-error flaw types (skipped words, restarted phrases) to an
existing dataset build, without touching the clips already there.

For every gold speech: one single-flaw clip per new type and severity, made
exactly as tools/build_dataset.py makes the others (same seeding scheme, same
gold alignment), plus mixed clips that combine a reading error with two
prosodic flaws. The manifest is updated in place.

Usage:  python tools/extend_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio  # noqa: E402
from speechlens.align import Word  # noqa: E402
from speechlens.inject import FLAW_TYPES, LEXICAL_TYPES, Injector  # noqa: E402
from tools.build_dataset import _seed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS, LIB = ROOT / "data/dataset", ROOT / "data/library"


def main():
    man = json.loads((DS / "manifest.json").read_text())
    have = {c["id"] for c in man["clips"]}
    new = []
    for sid in man["sources"]:
        y = audio.load(LIB / sid / "audio.wav")
        words = [Word(**w) for w in json.loads((LIB / sid / "alignment.json").read_text())]
        f0 = json.loads((LIB / sid / "meta.json").read_text())["f0_median_hz"]

        def emit(cid, out, nw, lb, kind, extra=None):
            audio.save(DS / "clips" / f"{cid}.wav", out)
            rec = {"id": cid, "source": sid, "kind": kind, "duration": round(len(out) / audio.SR, 2),
                   "labels": [l.to_dict() for l in lb], "max_severity": max([l.severity for l in lb], default=0)}
            rec |= extra or {}
            (DS / "labels" / f"{cid}.json").write_text(json.dumps(rec | {"words": [w.to_dict() for w in nw]}, indent=1))
            new.append(rec)

        for ft in LEXICAL_TYPES:
            for sev in range(1, 6):
                inj = Injector(y, words, f0, seed=_seed(sid, ft, sev, 0))
                out, nw, lb = inj.apply([(ft, sev)])
                if not lb:
                    print(f"  ! {sid} {ft} s{sev}: no placement")
                    continue
                emit(f"{sid}__{ft}_s{sev}", out, nw, lb, "single")
        # reading errors in context: one lexical + two prosodic flaws, moderate
        rng = np.random.default_rng(_seed(sid, "lexmix", 0))
        for k, sev in enumerate((2, 3, 4), start=1):
            types = [str(rng.choice(LEXICAL_TYPES))] + [str(t) for t in rng.choice(FLAW_TYPES, size=2, replace=False)]
            inj = Injector(y, words, f0, seed=int(rng.integers(2**31)))
            out, nw, lb = inj.apply([(t, sev) for t in types])
            emit(f"{sid}__lexmix{k}", out, nw, lb, "lexmix", {"lexical_mix": k})
        print(f"  {sid}: +{sum(c['source'] == sid for c in new)} clips")
    man["clips"] = [c for c in man["clips"] if c["id"] not in {n["id"] for n in new}] + new
    man["flaw_types"] = FLAW_TYPES + LEXICAL_TYPES
    (DS / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"added {len([n for n in new if n['id'] not in have])}, replaced {len([n for n in new if n['id'] in have])}")


if __name__ == "__main__":
    main()
