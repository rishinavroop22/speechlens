"""Manipulation check for the scripted human recordings.

A scripted flaw counts as *performed* only if a simple measurement over the
scripted words, against the same person's clean take A, reaches the
severity-1 level of the injection engine. The measurements are raw and
fixed in advance (they do not use the detector), so the check cannot favour
it. Results are written into each label as `performed` + `check`.

Usage: python tools/verify_human.py   (after tools/label_human.py)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio  # noqa: E402
from speechlens.contrast import _f0_spread  # noqa: E402
from speechlens.pipeline import analyze_cached  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data/dataset"

CRITERIA = {  # (measure, comparison, severity-1 level)
    "rushed": ("duration factor over the scripted sentence", "<=", 0.88),
    "dragging": ("duration factor over the scripted sentence", ">=", 1.12),
    "monotone": ("pitch range (p10-p90) ratio", "<=", 0.65),
    "trailing_off": ("loudness change over the second half, dB", "<=", -4.0),
    "mumbling": ("energy above 4 kHz ratio", "<=", 0.70),
    "pause_omission": ("reference pauses >= 180 ms cut below 40 %", ">=", 2),
    "awkward_pause": ("extra pause at the marked word, s", ">=", 0.35),
    "filler": ("time added at the marked boundary, s", ">=", 0.28),
    "stutter": ("time added before/in the marked word, s", ">=", 0.15),
}


def measure(t, ref, par, a, b):
    R, P = ref.wf, par.wf
    if t in ("rushed", "dragging"):
        return sum(P[i].dur for i in range(a, b + 1)) / max(1e-3, sum(R[i].dur for i in range(a, b + 1)))
    if t == "monotone":
        return _f0_spread(par, list(range(a, b + 1))) / max(0.1, _f0_spread(ref, list(range(a, b + 1))))
    if t == "trailing_off":
        n = b - a + 1
        return float(np.mean([P[i].int_mean_db - R[i].int_mean_db for i in range(a + n // 2, b + 1)]))
    if t == "mumbling":
        return float(np.nanmean([P[i].hiband4 / max(1e-5, R[i].hiband4) for i in range(a, b + 1)]))
    if t == "pause_omission":
        return sum(1 for i in range(a, b) if R[i].pause_after >= 0.18 and P[i].pause_after < 0.4 * R[i].pause_after)
    if t == "awkward_pause":
        return P[a].pause_after - R[a].pause_after
    if t == "filler":
        return (P[b].end - P[a].start) - (R[b].end - R[a].start)
    if t == "stutter":
        return (P[a].end - P[a - 1].end) - (R[a].end - R[a - 1].end)
    raise ValueError(t)


def main():
    man_p = DS / "manifest.json"
    man = json.loads(man_p.read_text())
    stats = {}
    for c in man["clips"]:
        if c["kind"] != "human":
            continue
        ref = analyze_cached(audio.load(DS / "clips" / f"{c['reference_clip']}.wav"), c["transcript"])
        par = analyze_cached(audio.load(DS / "clips" / f"{c['id']}.wav"), c["transcript"])
        for l in c["labels"]:
            name, op, lvl = CRITERIA[l["type"]]
            v = float(measure(l["type"], ref, par, l["word_start"], l["word_end"]))
            ok = v <= lvl if op == "<=" else v >= lvl
            l["performed"] = bool(ok)
            l["check"] = {"measure": name, "value": round(v, 3), "criterion": f"{op} {lvl}"}
            e = stats.setdefault(l["type"], [0, 0])
            e[0] += ok
            e[1] += 1
        lab = DS / "labels" / f"{c['id']}.json"
        d = json.loads(lab.read_text())
        d["labels"] = c["labels"]
        lab.write_text(json.dumps(d, indent=1))
    man_p.write_text(json.dumps(man, indent=1))
    tot = sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())
    print(f"performed as scripted: {tot[0]}/{tot[1]}")
    for t, (k, n) in stats.items():
        print(f"  {t:15s} {k}/{n}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/manipulation_check.json").write_text(json.dumps(
        {"criteria": {t: {"measure": m, "criterion": f"{o} {l}"} for t, (m, o, l) in CRITERIA.items()},
         "performed": {t: {"performed": k, "scripted": n} for t, (k, n) in stats.items()},
         "total": {"performed": tot[0], "scripted": tot[1]}}, indent=1))


if __name__ == "__main__":
    main()
