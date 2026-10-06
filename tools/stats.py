"""Statistics behind the headline numbers.

1. Bootstrap 95 % confidence intervals (2,000 resamples of whole clips) for the
   cross-validated synthetic P/R/F1, the human-recording precision/recall and
   the false-finding rate on controls.
2. Grounding F1 as the temporal-overlap criterion tightens (tIoU 0.1 .. 0.9).
3. Severity confusion matrix on matched flaws (true vs estimated level).
4. Feature evidence: for every flaw type, how strongly each word-level contrast
   signal separates the labelled words from the rest of the reading (Cohen's
   d), i.e. which measurements carry which flaw.

Usage:  python tools/stats.py   (after tools/evaluate.py and fit_thresholds.py)
Writes results/stats.json and docs/figures/{tiou_sweep,severity_confusion,feature_evidence}.svg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio  # noqa: E402
from speechlens.inject import ALL_TYPES  # noqa: E402
from tools.evaluate import match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS, LIB, OUT, FIG = ROOT / "data/dataset", ROOT / "data/library", ROOT / "results", ROOT / "docs/figures"
NAMES = {"rushed": "Rushed", "dragging": "Dragging", "monotone": "Monotone", "trailing_off": "Trailing off",
         "pause_omission": "Missing pauses", "awkward_pause": "Awkward pause", "filler": "Filler",
         "mumbling": "Muffled", "stutter": "Repeated onset", "skipped_word": "Skipped words", "restart": "Restart"}
RNG = np.random.default_rng(20261014)


def counts(items, thr=0.3):
    out = []
    for labels, preds in items:
        loc = [p for p in preds if p["scope"] == "local"]
        m = match(labels, loc, thr)
        out.append((len(m), len(loc) - len(m), len(labels) - len(m)))
    return np.array(out, float)


def prf(c):
    tp, fp, fn = c.sum(axis=0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def boot(c, fn, n=2000):
    vals = np.array([fn(c[RNG.integers(0, len(c), len(c))]) for _ in range(n)])
    lo, hi = np.percentile(vals, [2.5, 97.5], axis=0)
    return lo, hi


def ci_block(c):
    p, r, f = prf(c)
    lo, hi = boot(c, lambda x: np.array(prf(x)))
    return {"precision": [round(p, 3), round(lo[0], 3), round(hi[0], 3)],
            "recall": [round(r, 3), round(lo[1], 3), round(hi[1], 3)],
            "f1": [round(f, 3), round(lo[2], 3), round(hi[2], 3)], "n_clips": len(c)}


def word_signals(ref, par, pairs):
    """Per paired word: contrast signals with the reading's own median removed."""
    sig = {}
    R = [ref.wf[i] for i, _ in pairs]
    P = [par.wf[j] for _, j in pairs]
    d = lambda a: np.asarray(a, float) - np.nanmedian(a)  # noqa: E731
    sig["Duration (log2 ratio)"] = d([np.log2(max(p.dur, 1e-3) / max(r.dur, 1e-3)) for r, p in zip(R, P)])
    sig["Pitch range (log2 ratio)"] = d([np.log2(max(p.f0_range_st, 0.2) / max(r.f0_range_st, 0.2))
                                         if np.isfinite(p.f0_range_st) and np.isfinite(r.f0_range_st) else np.nan
                                         for r, p in zip(R, P)])
    sig["Loudness (dB)"] = d([p.int_mean_db - r.int_mean_db for r, p in zip(R, P)])
    sig["Energy > 2 kHz (log2)"] = d([np.log2(max(p.hiband, 1e-4) / max(r.hiband, 1e-4)) for r, p in zip(R, P)])
    sig["Pause after (s)"] = d([p.pause_after - r.pause_after for r, p in zip(R, P)])
    sig["Voicing in pause (s)"] = d([p.filled_after - r.filled_after for r, p in zip(R, P)])
    sig["Recogniser confidence"] = d([p.conf - r.conf for r, p in zip(R, P)])
    return sig


def feature_evidence(man):
    from speechlens.contrast import pair_words
    from speechlens.pipeline import analyze_cached
    acc = {t: {} for t in ALL_TYPES}
    refs = {}
    for c in man:
        if c["kind"] != "single" or c["labels"][0]["severity"] < 3:
            continue
        t = c["labels"][0]["type"]
        src = c["source"]
        text = (LIB / src / "transcript.txt").read_text()
        if src not in refs:
            refs[src] = analyze_cached(audio.load(LIB / src / "audio.wav"), text)
        ref = refs[src]
        par = analyze_cached(audio.load(audio.clip_path(DS / "clips", c["id"])), text)
        pairs = pair_words(ref, par)
        inside = np.zeros(len(pairs), bool)
        for lab in c["labels"]:
            for q, (_, j) in enumerate(pairs):
                w = par.words[j]
                if w.end > lab["start"] - 0.05 and w.start < lab["end"] + 0.05:
                    inside[q] = True
        for name, v in word_signals(ref, par, pairs).items():
            a = acc[t].setdefault(name, [[], []])
            a[0] += list(v[inside])
            a[1] += list(v[~inside])
    out = {}
    for t, sigs in acc.items():
        out[t] = {}
        for name, (ins, outs) in sigs.items():
            x, y = np.array(ins, float), np.array(outs, float)
            x, y = x[np.isfinite(x)], y[np.isfinite(y)]
            if len(x) < 3 or len(y) < 3:
                continue
            sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
            out[t][name] = round(float((x.mean() - y.mean()) / sp), 2) if sp > 0 else 0.0
    return out


def svg_heat(rows, cols, val, path, title, fmt="{:.1f}", diverging=False, vmax=None):
    cw, rh, lw, th = 74, 22, 120, 54
    w, h = lw + cw * len(cols) + 10, th + rh * len(rows) + 10
    vmax = vmax or max(abs(v) for r in rows for c_ in cols if (v := val(r, c_)) is not None) or 1
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Inter,Arial,sans-serif" font-size="10">',
           f'<text x="0" y="12" font-size="11" font-weight="700">{title}</text>']
    for k, c_ in enumerate(cols):
        x = lw + k * cw + cw / 2
        out.append(f'<text x="{x}" y="{th - 6}" text-anchor="middle" font-size="9">{c_}</text>')
    for i, r in enumerate(rows):
        y = th + i * rh
        out.append(f'<text x="{lw - 6}" y="{y + rh * 0.65}" text-anchor="end">{NAMES.get(r, r)}</text>')
        for k, c_ in enumerate(cols):
            v = val(r, c_)
            if v is None:
                continue
            a = min(1.0, abs(v) / vmax)
            col = (f"rgba(47,111,223,{a:.2f})" if (v >= 0 or not diverging) else f"rgba(184,52,74,{a:.2f})")
            out.append(f'<rect x="{lw + k * cw + 1}" y="{y + 1}" width="{cw - 2}" height="{rh - 2}" fill="{col}"/>')
            out.append(f'<text x="{lw + k * cw + cw / 2}" y="{y + rh * 0.65}" text-anchor="middle" '
                       f'fill="{"#fff" if a > 0.55 else "#1f2430"}">{fmt.format(v)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out))


def main():
    man = json.loads((DS / "manifest.json").read_text())["clips"]
    by = {c["id"]: c for c in man}
    cv = json.loads((OUT / "predictions_cv.json").read_text())
    ins = {r["id"]: r for r in json.loads((OUT / "predictions.json").read_text())}
    res = {}

    syn = [(by[r["id"]]["labels"], r["preds"]) for r in cv if r["kind"] in ("single", "mixed", "lexmix")]
    hum = [([l for l in by[r["id"]]["labels"] if l.get("performed", True)], r["preds"]) for r in cv if r["kind"] == "human"]
    ctrl = [r for r in cv if r["kind"] == "control"]
    res["ci_synthetic_cv"] = ci_block(counts(syn))
    res["ci_human_cv"] = ci_block(counts(hum))
    rate = np.array([[len(r["preds"]), by[r["id"]]["duration"] / 60] for r in ctrl])
    lo, hi = boot(rate, lambda x: x[:, 0].sum() / x[:, 1].sum())
    res["ci_controls_false_per_min"] = [round(rate[:, 0].sum() / rate[:, 1].sum(), 3), round(float(lo), 3), round(float(hi), 3)]

    # tIoU sweep on cross-validated synthetic predictions
    res["tiou_sweep"] = {f"{t:.1f}": round(prf(counts(syn, t))[2], 3) for t in np.arange(0.1, 0.95, 0.1)}

    # severity confusion on in-sample matched flaws (final tables)
    conf = np.zeros((5, 5), int)
    for c in man:
        if c["kind"] not in ("single", "mixed", "lexmix") or c["id"] not in ins:
            continue
        loc = [p for p in ins[c["id"]]["preds"] if p["scope"] == "local"]
        for i, j, _ in match(c["labels"], loc, 0.3):
            conf[c["labels"][i]["severity"] - 1, loc[j]["severity"] - 1] += 1
    res["severity_confusion"] = conf.tolist()
    res["severity_exact"] = round(float(np.trace(conf) / max(1, conf.sum())), 3)
    res["severity_within_1"] = round(float(sum(conf[i, j] for i in range(5) for j in range(5) if abs(i - j) <= 1)
                                           / max(1, conf.sum())), 3)

    res["feature_evidence"] = feature_evidence(man)
    OUT.mkdir(exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=1))

    # figures
    FIG.mkdir(parents=True, exist_ok=True)
    sw = res["tiou_sweep"]
    xs, ys = list(sw), list(sw.values())
    W, H = 360, 170
    pts = " ".join(f"{40 + i * (W - 60) / (len(xs) - 1):.1f},{H - 30 - v * (H - 50):.1f}" for i, v in enumerate(ys))
    lab = "".join(f'<text x="{40 + i * (W - 60) / (len(xs) - 1):.1f}" y="{H - 14}" text-anchor="middle">{x}</text>'
                  f'<text x="{40 + i * (W - 60) / (len(xs) - 1):.1f}" y="{H - 36 - v * (H - 50):.1f}" text-anchor="middle" font-size="8">{v:.2f}</text>'
                  for i, (x, v) in enumerate(zip(xs, ys)))
    (FIG / "tiou_sweep.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif" font-size="9">'
        f'<text x="0" y="11" font-size="11" font-weight="700">F1 vs required temporal overlap (tIoU)</text>'
        f'<line x1="40" y1="{H - 30}" x2="{W - 20}" y2="{H - 30}" stroke="#c8cfda"/>'
        f'<polyline points="{pts}" fill="none" stroke="#2f6fdf" stroke-width="2"/>{lab}</svg>')
    svg_heat([str(i) for i in range(1, 6)], [str(i) for i in range(1, 6)],
             lambda r, c_: int(conf[int(r) - 1, int(c_) - 1]) or None, FIG / "severity_confusion.svg",
             "Severity: true (rows) vs estimated (columns)", fmt="{:d}")
    fe = res["feature_evidence"]
    cols = list(next(iter(v for v in fe.values() if v)).keys())
    short = {c_: c_.split(" (")[0] for c_ in cols}
    svg_heat([t for t in ALL_TYPES if fe.get(t)], [short[c_] for c_ in cols],
             lambda r, c_: fe[r].get(next(k for k in cols if short[k] == c_)), FIG / "feature_evidence.svg",
             "Cohen's d: labelled words vs rest, per contrast signal", diverging=True, vmax=3.0)
    print(json.dumps({k: v for k, v in res.items() if k != "feature_evidence"}, indent=1))


if __name__ == "__main__":
    main()
