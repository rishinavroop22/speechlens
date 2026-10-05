"""Fit severity tables from data, with leave-one-speech-out validation.

For every flaw type, the detector's raw magnitude (e.g. local duration factor
for "rushed") is collected on regions matched to ground truth. The table entry
for severity k is the median magnitude of true-severity-k matches (made
monotone). Severity error is reported for each speech using tables fitted on
the OTHER speeches only; the final tables, fitted on all speeches, are written
to speechlens/calibration.json and used at inference.

Usage: python tools/calibrate.py   (after tools/evaluate.py)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import match  # noqa: E402
from speechlens.contrast import _sev  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = {  # (table, increasing) - the injection parameter tables
    "rushed": ([0.88, 0.80, 0.72, 0.64, 0.55], False), "dragging": ([1.12, 1.24, 1.36, 1.48, 1.62], True),
    "monotone": ([0.35, 0.55, 0.70, 0.85, 0.97], True), "trailing_off": ([4, 7, 10, 14, 18], True),
    "pause_omission": ([0.60, 0.45, 0.30, 0.15, 0.04], False), "awkward_pause": ([0.35, 0.60, 0.90, 1.30, 1.80], True),
    "filler": ([0.28, 0.36, 0.45, 0.55, 0.65], True), "mumbling": ([0.8, 1.6, 2.6, 3.6, 5.0], True),
    "stutter": ([1, 1.5, 2, 2.5, 3], True),
}


def collect():
    man = {c["id"]: c for c in json.loads((ROOT / "data/dataset/manifest.json").read_text())["clips"]}
    preds = {r["id"]: r for r in json.loads((ROOT / "results/predictions.json").read_text())}
    rows = []  # (speech, type, magnitude, true severity)
    for cid, c in man.items():
        if c["kind"] not in ("single", "mixed") or cid not in preds:
            continue
        local = [p for p in preds[cid]["preds"] if p["scope"] == "local"]
        for i, j, _ in match(c["labels"], local, 0.3):
            if local[j].get("magnitude") is not None:
                rows.append((c["source"], c["labels"][i]["type"], local[j]["magnitude"], c["labels"][i]["severity"]))
    return rows


def fit(rows):
    by = defaultdict(lambda: defaultdict(list))
    for _, t, m, s in rows:
        by[t][s].append(m)
    out = {}
    for t, (tbl, inc) in DEFAULT.items():
        new = list(tbl)
        for k in range(1, 6):
            if len(by[t][k]) >= 2:
                new[k - 1] = float(np.median(by[t][k]))
        # enforce monotone direction
        arr = np.array(new)
        arr = np.maximum.accumulate(arr) if inc else np.minimum.accumulate(arr)
        out[t] = {"table": [round(float(v), 4) for v in arr], "increasing": inc,
                  "n": {k: len(v) for k, v in sorted(by[t].items())}}
    return out


def main():
    rows = collect()
    speeches = sorted({r[0] for r in rows})
    err_default, err_loso = [], []
    for sp in speeches:
        cal = fit([r for r in rows if r[0] != sp])
        for s_, t, m, true in rows:
            if s_ != sp:
                continue
            err_default.append(_sev(m, *DEFAULT[t]) - true)
            err_loso.append(_sev(m, cal[t]["table"], cal[t]["increasing"]) - true)
    summ = lambda e: {"mae": round(float(np.mean(np.abs(e))), 3), "within_1": round(float(np.mean(np.abs(e) <= 1)), 3),
                      "bias": round(float(np.mean(e)), 3), "n": len(e)}
    report = {"default_tables": summ(err_default), "calibrated_leave_one_speech_out": summ(err_loso)}
    final = fit(rows)
    (ROOT / "speechlens/calibration.json").write_text(json.dumps({t: {k: v[k] for k in ("table", "increasing")}
                                                                  for t, v in final.items()}, indent=1))
    ev = ROOT / "results/evaluation.json"
    if ev.exists():
        e = json.loads(ev.read_text())
        e["severity_calibration"] = report | {"tables": final}
        ev.write_text(json.dumps(e, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
