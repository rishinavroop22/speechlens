"""Evaluate SpeechLens against the dataset's ground-truth labels.

For every clip the *full* pipeline runs on raw audio (forced alignment of the
flawed recording included - no oracle timings). Reports:
  * temporal grounding: precision / recall / F1 at tIoU >= 0.3 and 0.5,
    mean tIoU and boundary error of matched regions, per flaw type;
  * sensitivity by severity (does detection rate rise 1 -> 5?);
  * severity estimation error;
  * false positives on controls (identity, pitch +/-4 st, -12 dB);
  * score validity: Spearman correlation of overall score with flaw load;
  * reproducibility: a second run over a subset must give identical output.

Usage:  python tools/evaluate.py [--limit N] [--workers 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio  # noqa: E402
from speechlens.contrast import DIMENSION  # noqa: E402
from speechlens.inject import FLAW_TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS, LIB, OUT = ROOT / "data/dataset", ROOT / "data/library", ROOT / "results"


def run_clip(clip):
    from speechlens.pipeline import evaluate
    src = LIB / clip["source"]
    rep = evaluate(audio.load(src / "audio.wav"), (src / "transcript.txt").read_text(),
                   audio.load(DS / "clips" / f"{clip['id']}.wav"), None, "declamation")
    preds = [{k: r[k] for k in ("type", "start", "end", "severity", "scope", "confidence", "magnitude")} for r in rep["regions"]]
    return {"id": clip["id"], "preds": preds, "score": rep["score"], "align_method": rep["align_method"]}


def tiou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def match(truth, preds, thr, by="type"):
    """Greedy one-to-one matching by tIoU, same type (or same dimension)."""
    key = (lambda r: r["type"]) if by == "type" else (lambda r: DIMENSION[r["type"]])
    pairs = sorted(((tiou(t, p), i, j) for i, t in enumerate(truth) for j, p in enumerate(preds)
                    if key(t) == key(p)), reverse=True)
    used_t, used_p, out = set(), set(), []
    for iou, i, j in pairs:
        if iou >= thr and i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            out.append((i, j, iou))
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return round(float(r.statistic), 3), float(r.pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    man = json.loads((DS / "manifest.json").read_text())
    clips = man["clips"][: args.limit] if args.limit else man["clips"]
    import os
    os.environ.setdefault("SPEECHLENS_THREADS", str(max(1, (os.cpu_count() or 1) // args.workers)))
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(run_clip, clips, chunksize=2))
    by_id = {r["id"]: r for r in results}

    # reproducibility: rerun a subset in a fresh process pool, compare bytes
    subset = clips[:: max(1, len(clips) // 8)][:8]
    with ProcessPoolExecutor(1) as ex:
        again = list(ex.map(run_clip, subset))
    h = lambda r: hashlib.sha1(json.dumps(r, sort_keys=True).encode()).hexdigest()
    repro = all(h(a) == h(by_id[a["id"]]) for a in again)

    report = {"n_clips": len(clips), "align_method": results[0]["align_method"] if results else None,
              "reproducible": repro, "reproducibility_subset": len(subset)}
    flawed = [c for c in clips if c["labels"] and c["kind"] != "human"]
    human = [c for c in clips if c["kind"] == "human"]

    def grounding(cs, thr):
        rows = {}
        for ft in FLAW_TYPES + ["__all__"]:
            tp = fp = fn = 0
            ious, berr = [], []
            for c in cs:
                local = [p for p in by_id[c["id"]]["preds"] if p["scope"] == "local"]
                T = [l for l in c["labels"] if ft in ("__all__", l["type"])]
                Pp = [p for p in local if ft in ("__all__", p["type"])]
                m = match(T, Pp, thr)
                tp += len(m)
                fp += len(Pp) - len(m)
                fn += len(T) - len(m)
                for i, j, iou in m:
                    ious.append(iou)
                    berr += [abs(T[i]["start"] - Pp[j]["start"]), abs(T[i]["end"] - Pp[j]["end"])]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            rows[ft] = {"precision": round(p, 3), "recall": round(r, 3),
                        "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
                        "mean_tiou": round(float(np.mean(ious)), 3) if ious else None,
                        "boundary_err_ms": round(float(np.median(berr)) * 1000) if berr else None,
                        "tp": tp, "fp": fp, "fn": fn}
        return rows

    report["grounding_tiou_0.3"] = grounding(flawed, 0.3)
    report["grounding_tiou_0.5"] = grounding(flawed, 0.5)
    if human:
        report["human_grounding_tiou_0.3"] = grounding(human, 0.3)

    # sensitivity by severity (single-flaw clips, tIoU >= 0.3)
    sens = {}
    for ft in FLAW_TYPES:
        row = []
        for sev in range(1, 6):
            cs = [c for c in flawed if c["kind"] == "single" and c["labels"] and c["labels"][0]["type"] == ft
                  and c["labels"][0]["severity"] == sev]
            hit = tot = 0
            for c in cs:
                local = [p for p in by_id[c["id"]]["preds"] if p["scope"] == "local"]
                hit += len(match(c["labels"], local, 0.3))
                tot += len(c["labels"])
            row.append(round(hit / tot, 3) if tot else None)
        sens[ft] = row
    report["recall_by_severity"] = sens

    # severity estimation on matched regions
    err = []
    for c in flawed:
        local = [p for p in by_id[c["id"]]["preds"] if p["scope"] == "local"]
        for i, j, _ in match(c["labels"], local, 0.3):
            err.append(local[j]["severity"] - c["labels"][i]["severity"])
    report["severity"] = {"mae": round(float(np.mean(np.abs(err))), 3) if err else None,
                          "within_1": round(float(np.mean(np.abs(err) <= 1)), 3) if err else None,
                          "bias": round(float(np.mean(err)), 3) if err else None, "n": len(err)}

    # controls
    ctrl = {}
    for c in clips:
        if c["kind"] != "control":
            continue
        k = c.get("control", "control")
        preds = by_id[c["id"]]["preds"]
        e = ctrl.setdefault(k, {"clips": 0, "false_regions": 0, "minutes": 0.0, "scores": []})
        e["clips"] += 1
        e["false_regions"] += len(preds)
        e["minutes"] += c["duration"] / 60
        e["scores"].append(by_id[c["id"]]["score"]["overall"])
    for k, e in ctrl.items():
        e["false_per_min"] = round(e["false_regions"] / max(e["minutes"], 1e-9), 3)
        e["mean_score"] = round(float(np.mean(e.pop("scores"))), 1)
        e["minutes"] = round(e["minutes"], 2)
    report["controls"] = ctrl

    # score validity
    load = [sum(l["severity"] for l in c["labels"]) for c in clips if c["kind"] != "human"]
    sc = [by_id[c["id"]]["score"]["overall"] for c in clips if c["kind"] != "human"]
    rho, p = spearman(load, sc) if len(set(load)) > 1 else (None, None)
    mix = [c for c in clips if c["kind"] == "mixed"]
    mrho, mp = spearman([c["mix_level"] for c in mix], [by_id[c["id"]]["score"]["overall"] for c in mix]) if mix else (None, None)
    per_type = {}
    for ft in FLAW_TYPES:
        cs = [c for c in clips if c["kind"] == "single" and c["labels"][0]["type"] == ft]
        if len(cs) > 3:
            per_type[ft] = spearman([c["labels"][0]["severity"] for c in cs], [by_id[c["id"]]["score"]["overall"] for c in cs])[0]
    report["score_validity"] = {"spearman_load_vs_score": rho, "p": p, "spearman_mix_level_vs_score": mrho, "p_mix": mp,
                                "spearman_severity_vs_score_by_type": per_type,
                                "mean_score_by_mix_level": {lvl: round(float(np.mean([by_id[c["id"]]["score"]["overall"]
                                                                                    for c in mix if c["mix_level"] == lvl])), 1)
                                                            for lvl in sorted({c["mix_level"] for c in mix})}}
    OUT.mkdir(exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=1))
    (OUT / "predictions.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: report[k] for k in ("n_clips", "reproducible", "severity", "controls", "score_validity")}, indent=1))
    g = report["grounding_tiou_0.3"]
    print("\ntype              P     R     F1   tIoU  bnd(ms)")
    for ft, r in g.items():
        print(f"{ft:15s} {r['precision']:.2f}  {r['recall']:.2f}  {r['f1']:.2f}  {r['mean_tiou'] or 0:.2f}  {r['boundary_err_ms']}")
    print("\nrecall by severity 1..5")
    for ft, row in sens.items():
        print(f"{ft:15s}", row)


if __name__ == "__main__":
    main()
