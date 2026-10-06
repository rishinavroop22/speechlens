"""Fit per-flaw-type significance gates, with honest cross-validation.

Every finding carries a raw magnitude (duration factor, extra pause seconds,
dB drop, ...). A gate keeps a finding only when its magnitude is beyond a
per-type threshold, so deviations within natural take-to-take variation are
not reported. The threshold for each type maximises F1 on the fitting data
(synthetic clips + human takes, each source weighted equally).

Validation, never testing on data used for fitting:
  * human takes: leave-one-speaker-out (fit without the tested recorder);
  * synthetic clips: leave-one-speech-out (fit without that speech's
    synthetic clips AND without human takes of that passage).
Final gates (fit on everything) are written to speechlens/calibration.json.

Requires predictions made WITHOUT gates:
  SPEECHLENS_NO_GATE=1 python tools/evaluate.py && python tools/fit_thresholds.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import human_view, match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["rushed", "dragging", "monotone", "trailing_off", "pause_omission", "awkward_pause", "filler", "mumbling", "stutter"]
WORSE = {"rushed": "lower", "pause_omission": "lower"}  # all others: higher magnitude = worse
# never gate above the severity-3 injection parameter: a moderate flaw must stay detectable
CAP = {"rushed": 0.72, "dragging": 1.36, "monotone": 0.70, "trailing_off": 10.0, "pause_omission": 0.30,
       "awkward_pause": 0.90, "filler": 0.45, "mumbling": 2.6, "stutter": 2.0}


def keep(p, gates):
    g = gates.get(p["type"])
    if g is None or p["scope"] != "local" or p.get("magnitude") is None:
        return True
    return p["magnitude"] >= g["min"] if g["worse"] == "higher" else p["magnitude"] <= g["min"]


def items(clips, preds):
    """Per clip: labels and its local predictions (with magnitudes)."""
    return [(c, [p for p in preds[c["id"]]["preds"] if p["scope"] == "local"]) for c in clips]


def counts(data, ftype, gates):
    tp = fp = fn = 0
    for c, local in data:
        P = [p for p in local if p["type"] == ftype and keep(p, gates)]
        T = [l for l in c["labels"] if l["type"] == ftype]
        m = len(match(T, P, 0.3))
        tp += m
        fp += len(P) - m
        fn += len(T) - m
    return tp, fp, fn


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def fit(syn, hum):
    gates = {}
    for t in TYPES:
        worse = WORSE.get(t, "higher")
        mags = sorted({round(p["magnitude"], 4) for _, L in syn + hum for p in L
                       if p["type"] == t and p.get("magnitude") is not None})
        if worse == "higher":
            cands = [m for m in mags if m <= CAP[t]]
        else:
            cands = [m for m in mags if m >= CAP[t]]
        best, best_f = None, -1.0
        base = {}
        for m in [None] + cands:
            g = dict(base) if m is None else {t: {"min": m, "worse": worse}}
            fs = f1(*counts(syn, t, g))
            fh = f1(*counts(hum, t, g)) if any(l["type"] == t for c, _ in hum for l in c["labels"]) else fs
            score = 0.5 * fs + 0.5 * fh
            if score > best_f + 1e-9:
                best, best_f = m, score
        if best is not None:
            gates[t] = {"min": best, "worse": worse}
    return gates


def table(data, gates):
    rows = {}
    for t in TYPES + ["__all__"]:
        tp = fp = fn = 0
        for tt in (TYPES if t == "__all__" else [t]):
            a, b, c = counts(data, tt, gates)
            tp, fp, fn = tp + a, fp + b, fn + c
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows[t] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1(tp, fp, fn), 3),
                   "tp": tp, "fp": fp, "fn": fn}
    return rows


def fp_per_min(data, gates):
    n = sum(1 for c, L in data for p in L if keep(p, gates))
    mins = sum(c["duration"] for c, _ in data) / 60
    return round(n / mins, 3) if mins else None


def main():
    man = json.loads((ROOT / "data/dataset/manifest.json").read_text())["clips"]
    preds = {r["id"]: r for r in json.loads((ROOT / "results/predictions.json").read_text())}
    syn = items([c for c in man if c["kind"] in ("single", "mixed")], preds)
    ctrl = items([c for c in man if c["kind"] == "control"], preds)
    hum_all = items([c for c in man if c["kind"] == "human"], preds)
    hum = []  # verified view: only labels that passed the manipulation check
    for c, L in hum_all:
        labels, L2 = human_view(c, L)
        hum.append((dict(c, labels=labels), L2))
    speakers = sorted({c["speaker"] for c, _ in hum})
    speeches = sorted({c["source"] for c, _ in syn})

    # human: leave one speaker out
    cv_h = []
    for s in speakers:
        g = fit(syn, [x for x in hum if x[0]["speaker"] != s])
        cv_h += [(c, [p for p in L if keep(p, g)]) for c, L in hum if c["speaker"] == s]
    # synthetic + controls: leave one speech out
    cv_s, cv_c = [], []
    for sp in speeches:
        g = fit([x for x in syn if x[0]["source"] != sp], [x for x in hum if x[0]["source"] != sp])
        cv_s += [(c, [p for p in L if keep(p, g)]) for c, L in syn if c["source"] == sp]
        cv_c += [(c, [p for p in L if keep(p, g)]) for c, L in ctrl if c["source"] == sp]

    final = fit(syn, hum)
    report = {
        "method": "per-type magnitude gates maximising F1 (synthetic and human weighted equally); "
                  "human results leave-one-speaker-out, synthetic leave-one-speech-out",
        "cv_human": table(cv_h, {}), "cv_synthetic": table(cv_s, {}),
        "cv_controls_false_per_min": fp_per_min(cv_c, {}),
        "cv_human_by_speaker": {s: table([x for x in cv_h if x[0]["speaker"] == s], {})["__all__"] for s in speakers},
        "ungated_human": table(hum, {}), "ungated_synthetic": table(syn, {}),
        "cv_human_all_scripted": None,
        "gates": final,
    }
    path = ROOT / "speechlens/calibration.json"
    cal = json.loads(path.read_text()) if path.exists() else {}
    cal["_gates"] = final
    path.write_text(json.dumps(cal, indent=1))
    ev = ROOT / "results/evaluation.json"
    e = json.loads(ev.read_text())
    e["cross_validated"] = report
    ev.write_text(json.dumps(e, indent=1))
    for name in ("cv_synthetic", "cv_human", "ungated_human"):
        r = report[name]
        print(f"\n{name}:  F1 {r['__all__']['f1']}  P {r['__all__']['precision']}  R {r['__all__']['recall']}")
        for t in TYPES:
            print(f"  {t:15s} P {r[t]['precision']:.2f} R {r[t]['recall']:.2f} F1 {r[t]['f1']:.2f}  ({r[t]['tp']}/{r[t]['tp'] + r[t]['fn']})")
    print("\ncontrols false/min (CV):", report["cv_controls_false_per_min"])
    print("by speaker:", {k: v["f1"] for k, v in report["cv_human_by_speaker"].items()})
    print("gates:", json.dumps(final))


if __name__ == "__main__":
    main()
