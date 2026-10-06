"""Interventional test of the explanations.

For every severity-3 and -4 single-flaw clip where SpeechLens found the
injected flaw, apply the counterfactual repair (speechlens/counterfactual.py)
to that finding - an edit of exactly the variable the explanation names -
then run the full pipeline again on the repaired audio. If the explanation
named the true cause, the finding should disappear, no new finding should
appear in its place, and the score should move back toward the clean
reading's score.

Usage:  python tools/counterfactual_eval.py [--workers 2] [--quick]
Writes results/counterfactual.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio  # noqa: E402
from tools.evaluate import match, tiou  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS, LIB, OUT = ROOT / "data/dataset", ROOT / "data/library", ROOT / "results"


def run(clip):
    from speechlens.contrast import compare
    from speechlens.counterfactual import repair
    from speechlens.explain import load_rubric, score
    from speechlens.pipeline import analyze_cached
    src = LIB / clip["source"]
    text = (src / "transcript.txt").read_text()
    ref = analyze_cached(audio.load(src / "audio.wav"), text)
    y = audio.load(audio.clip_path(DS / "clips", clip["id"]))
    par = analyze_cached(y, text)
    rub = load_rubric("declamation")
    before = compare(ref, par)
    sc0 = score(before["regions"], len(par.words), par.duration, rub)["overall"]
    loc = [r for r in before["regions"] if r.scope == "local"]
    m = match(clip["labels"], [{"type": r.type, "start": r.start, "end": r.end} for r in loc], 0.3)
    if not m:
        return {"id": clip["id"], "type": clip["labels"][0]["type"], "found": False}
    reg = loc[m[0][1]]
    try:
        fixed, info = repair(y, ref, par, reg.to_dict(), before["global"])
    except ValueError as e:
        return {"id": clip["id"], "type": reg.type, "found": True, "repaired": False, "error": str(e)}
    par2 = analyze_cached(fixed, text)
    after = compare(ref, par2)
    sc1 = score(after["regions"], len(par2.words), par2.duration, rub)["overall"]
    # the same finding after the repair: same type, overlapping the edited span
    s, e = reg.start, reg.end + info["shift"]
    same = [r for r in after["regions"] if r.scope == "local" and r.type == reg.type
            and tiou({"start": s, "end": e}, {"start": r.start, "end": r.end}) > 0.1]
    near_new = [r.type for r in after["regions"] if r.scope == "local" and r.type != reg.type
                and min(e + 1, r.end) - max(s - 1, r.start) > 0
                and not any(o.type == r.type and min(o.end, r.end) - max(o.start, r.start) > 0 for o in before["regions"])]
    gold = analyze_cached(audio.load(src / "audio.wav"), text)
    sc_gold = score(compare(ref, gold)["regions"], len(gold.words), gold.duration, rub)["overall"]
    return {"id": clip["id"], "type": reg.type, "severity": clip["labels"][0]["severity"], "found": True,
            "repaired": True, "edit": info["edit"], "gone": not same,
            "residual_severity": max((r.severity for r in same), default=0), "before_severity": reg.severity,
            "new_nearby": near_new, "score_before": sc0, "score_after": sc1, "score_gold": sc_gold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    man = json.loads((DS / "manifest.json").read_text())
    clips = [c for c in man["clips"] if c["kind"] == "single" and c["labels"][0]["severity"] in (3, 4)]
    if args.quick:
        clips = [c for c in clips if c["source"] == "jfk" and c["labels"][0]["severity"] == 4]
    os.environ.setdefault("SPEECHLENS_THREADS", str(max(1, (os.cpu_count() or 1) // args.workers)))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run, clips, chunksize=1))
    rep = [r for r in rows if r.get("repaired")]
    by = {}
    for r in rep:
        e = by.setdefault(r["type"], {"n": 0, "gone": 0, "recovered": []})
        e["n"] += 1
        e["gone"] += r["gone"]
        gap = r["score_gold"] - r["score_before"]
        if gap > 0:
            e["recovered"].append((r["score_after"] - r["score_before"]) / gap)
    gaps = [(r["score_after"] - r["score_before"]) / (r["score_gold"] - r["score_before"])
            for r in rep if r["score_gold"] - r["score_before"] > 0]
    summary = {
        "clips": len(clips), "found": sum(r["found"] for r in rows), "repaired": len(rep),
        "finding_removed": round(sum(r["gone"] for r in rep) / max(1, len(rep)), 3),
        "severity_reduced_or_removed": round(sum(r["residual_severity"] < r["before_severity"] for r in rep)
                                             / max(1, len(rep)), 3),
        "new_finding_nearby": round(sum(bool(r["new_nearby"]) for r in rep) / max(1, len(rep)), 3),
        "mean_score_before": round(float(np.mean([r["score_before"] for r in rep])), 1) if rep else None,
        "mean_score_after": round(float(np.mean([r["score_after"] for r in rep])), 1) if rep else None,
        "median_gap_recovered": round(float(np.median(gaps)), 3) if gaps else None,
        "by_type": {k: {"n": v["n"], "removed": round(v["gone"] / v["n"], 2),
                        "median_gap_recovered": round(float(np.median(v["recovered"])), 2) if v["recovered"] else None}
                    for k, v in sorted(by.items())},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "counterfactual.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
