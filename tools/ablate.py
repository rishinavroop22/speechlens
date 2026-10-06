"""Ablations and a reference-free baseline: which parts of the method carry
the result?

Every variant reruns the contrast engine on the same cached analyses (same
alignment, same features) with one component switched off, see
speechlens/contrast.py:_ablations. The reference-free baseline (self_ref)
keeps the whole detector but replaces the reference reading with the
participant's own running median, i.e. it looks for spots that are unusual
*within* the reading - the standard approach when no reference exists.

Reported per variant: grounding P/R/F1 at tIoU >= 0.3 on the synthetic clips,
false findings per minute on the controls, Spearman(mix level, score), and
the verified human-take precision/recall.

Usage:  python tools/ablate.py [--workers 2]
Writes results/ablation.json.
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
from tools.evaluate import human_view, match, spearman  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS, LIB, OUT = ROOT / "data/dataset", ROOT / "data/library", ROOT / "results"

VARIANTS = {
    "full": {"ablate": "", "label": "SpeechLens (full)"},
    "no_gate": {"ablate": "", "no_gate": True, "label": "Without data-fitted gates"},
    "no_rules": {"ablate": "no_rules", "label": "Without conflict rules"},
    "no_trend": {"ablate": "no_trend", "label": "Without local boundary trend"},
    "no_global": {"ablate": "no_global", "label": "Without habit / local split"},
    "no_spread": {"ablate": "no_spread", "label": "Without speaker-relative scale"},
    "self_ref": {"ablate": "self_ref", "label": "Reference-free baseline"},
    "self_ref_nogate": {"ablate": "self_ref", "no_gate": True, "label": "Reference-free, no gates"},
}
ORDER = list(VARIANTS) + ["self_ref_oracle"]


def run(job):
    clip, var = job
    v = VARIANTS[var]
    os.environ["SPEECHLENS_ABLATE"] = v["ablate"]
    if v.get("no_gate"):
        os.environ["SPEECHLENS_NO_GATE"] = "1"
    else:
        os.environ.pop("SPEECHLENS_NO_GATE", None)
    from speechlens.contrast import compare
    from speechlens.explain import load_rubric, score
    from speechlens.pipeline import analyze_cached
    src = LIB / clip["source"]
    if clip.get("reference_clip"):
        ref_y, ref_t = audio.load(audio.clip_path(DS / "clips", clip["reference_clip"])), clip["transcript"]
    else:
        ref_y, ref_t = audio.load(src / "audio.wav"), (src / "transcript.txt").read_text()
    ref = analyze_cached(ref_y, ref_t)
    par = analyze_cached(audio.load(audio.clip_path(DS / "clips", clip["id"])), clip.get("transcript") or ref_t)
    res = compare(ref, par)
    sc = score(res["regions"], len(par.words), par.duration, load_rubric("declamation"))
    return {"id": clip["id"], "var": var, "score": sc["overall"],
            "preds": [{"type": r.type, "start": r.start, "end": r.end, "severity": r.severity, "scope": r.scope,
                       "magnitude": r.magnitude} for r in res["regions"]]}


def summarize(clips, rows):
    by = {r["id"]: r for r in rows}
    fl = [c for c in clips if c["kind"] in ("single", "mixed")]
    tp = fp = fn = 0
    for c in fl:
        loc = [p for p in by[c["id"]]["preds"] if p["scope"] == "local"]
        m = match(c["labels"], loc, 0.3)
        tp, fp, fn = tp + len(m), fp + len(loc) - len(m), fn + len(c["labels"]) - len(m)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    ctrl = [c for c in clips if c["kind"] == "control"]
    mins = sum(c["duration"] for c in ctrl) / 60
    mix = [c for c in clips if c["kind"] == "mixed"]
    rho = spearman([c["mix_level"] for c in mix], [by[c["id"]]["score"] for c in mix])[0]
    htp = hfp = hfn = 0
    for c in [c for c in clips if c["kind"] == "human"]:
        loc = [q for q in by[c["id"]]["preds"] if q["scope"] == "local"]
        labels, loc = human_view(c, loc)
        m = match(labels, loc, 0.3)
        htp, hfp, hfn = htp + len(m), hfp + len(loc) - len(m), hfn + len(labels) - len(m)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "controls_false_per_min": round(sum(len(by[c["id"]]["preds"]) for c in ctrl) / mins, 3),
            "spearman_mix_vs_score": rho,
            "human_precision": round(htp / (htp + hfp), 3) if htp + hfp else 0.0,
            "human_recall": round(htp / (htp + hfn), 3) if htp + hfn else 0.0}


def oracle_tuned(clips, rows):
    """Best case for a variant: per flaw type, the magnitude cut-off that
    maximises F1 *on the evaluation labels themselves* (an upper bound - no
    real system gets to tune on its test set)."""
    from speechlens.contrast import GATES
    from speechlens.inject import FLAW_TYPES
    by = {r["id"]: r for r in rows}
    fl = [c for c in clips if c["kind"] in ("single", "mixed")]
    ctrl = [c for c in clips if c["kind"] == "control"]
    worse = {t: GATES.get(t, {}).get("worse", "higher") for t in FLAW_TYPES}
    keep_fn = lambda p, cut: p["magnitude"] is None or cut is None or (
        p["magnitude"] >= cut if worse[p["type"]] == "higher" else p["magnitude"] <= cut)
    cuts = {}
    for t in FLAW_TYPES:
        mags = sorted({p["magnitude"] for c in fl for p in by[c["id"]]["preds"]
                       if p["type"] == t and p["scope"] == "local" and p["magnitude"] is not None})
        best = (-1, None)
        for cut in [None] + mags[:: max(1, len(mags) // 60)]:
            tp = fp = fn = 0
            for c in fl:
                loc = [p for p in by[c["id"]]["preds"] if p["scope"] == "local" and p["type"] == t and keep_fn(p, cut)]
                lab = [l for l in c["labels"] if l["type"] == t]
                m = match(lab, loc, 0.3)
                tp, fp, fn = tp + len(m), fp + len(loc) - len(m), fn + len(lab) - len(m)
            f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f > best[0]:
                best = (f, cut)
        cuts[t] = best[1]
    tuned = [dict(r, preds=[p for p in r["preds"] if p["scope"] != "local" or keep_fn(p, cuts[p["type"]])])
             for r in rows]
    out = summarize(clips, tuned)
    out["spearman_mix_vs_score"] = None   # scores are not recomputed after filtering
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    man = json.loads((DS / "manifest.json").read_text())
    clips = [c for c in man["clips"] if c["kind"] in ("single", "mixed", "control", "human")]
    vars_ = args.only or list(VARIANTS)
    prev = json.loads((OUT / "ablation.json").read_text()) if (OUT / "ablation.json").exists() else {"variants": {}}
    os.environ.setdefault("SPEECHLENS_THREADS", "1")
    for var in vars_:
        with ProcessPoolExecutor(args.workers) as ex:  # fresh workers: env is read per process
            rows = list(ex.map(run, [(c, var) for c in clips], chunksize=4))
        prev["variants"][var] = dict(summarize(clips, rows), label=VARIANTS[var]["label"])
        if var == "self_ref_nogate":
            prev["variants"]["self_ref_oracle"] = dict(oracle_tuned(clips, rows),
                                                       label="Reference-free, thresholds tuned on test labels")
        shown = [var] + (["self_ref_oracle"] if var == "self_ref_nogate" else [])
        for s in [prev["variants"][k] for k in shown]:
            print(f"{s['label']:28s} P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} "
                  f"ctrl/min {s['controls_false_per_min']:.3f} rho {s['spearman_mix_vs_score']} "
                  f"human P {s['human_precision']:.3f} R {s['human_recall']:.3f}", flush=True)
        prev["variants"] = {k: prev["variants"][k] for k in ORDER if k in prev["variants"]}
        (OUT / "ablation.json").write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()
