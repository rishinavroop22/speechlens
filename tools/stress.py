"""Stress test: how does grounding hold up when the participant's recording
is made under worse conditions than the reference?

The reference stays the clean studio speech; only the participant audio is
degraded, which is the realistic case (a phone recording in a noisy room,
judged against a good reading). Conditions:

  babble noise   SNR 20 / 10 / 5 dB - other speakers from the library mixed in
  white noise    SNR 10 dB
  reverberation  RT60 0.4 s / 0.8 s - synthetic exponential room response
  telephone      8 kHz, 300-3400 Hz band-pass
  mp3            32 kbit/s, decoded again
  tempo          whole reading 10 % faster / slower (PSOLA), labels rescaled -
                 a uniform habit, which must not be reported as local flaws

Clips: every severity-4 single-flaw clip (9 types x 5 speeches), the level-6
mixed clips, and each speech's clean gold copy as a null (any finding on a
degraded copy of the reference itself is false). Everything is seeded per
clip and condition, so the run is reproducible.

Usage:  python tools/stress.py [--workers 2] [--quick]
Writes results/stress.json and docs/figures/stress.png.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio  # noqa: E402
from speechlens.inject import FLAW_TYPES  # noqa: E402
from tools.evaluate import match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS, LIB, OUT = ROOT / "data/dataset", ROOT / "data/library", ROOT / "results"
SR = audio.SR
SOURCES = ["fdr", "jfk", "lbj", "obama", "reagan"]

CONDITIONS = {
    "clean": {},
    "babble_20db": {"noise": "babble", "snr": 20},
    "babble_10db": {"noise": "babble", "snr": 10},
    "babble_5db": {"noise": "babble", "snr": 5},
    "white_10db": {"noise": "white", "snr": 10},
    "reverb_0.4s": {"rt60": 0.4},
    "reverb_0.8s": {"rt60": 0.8},
    "telephone": {"telephone": True},
    "mp3_32k": {"mp3": 32},
    "tempo_0.9x": {"tempo": 1 / 0.9},   # duration factor: 10 % slower
    "tempo_1.1x": {"tempo": 1 / 1.1},   # 10 % faster
}
# mitigation: the same noisy conditions with the reference put through the
# participant's estimated noise channel (speechlens/channel.py)
MATCHED = ["babble_20db", "babble_10db", "white_10db"]
LABEL = {"clean": "Clean", "babble_20db": "Babble 20 dB", "babble_10db": "Babble 10 dB", "babble_5db": "Babble 5 dB",
         "white_10db": "White noise 10 dB", "reverb_0.4s": "Reverb 0.4 s", "reverb_0.8s": "Reverb 0.8 s",
         "telephone": "Telephone band", "mp3_32k": "MP3 32 kbit/s", "tempo_0.9x": "Tempo -10 %",
         "tempo_1.1x": "Tempo +10 %"}
for _c in MATCHED:
    CONDITIONS[_c + "+match"] = CONDITIONS[_c]
    LABEL[_c + "+match"] = LABEL[_c] + ", channel-matched"


def _rng(*keys):
    return np.random.default_rng(zlib.crc32("|".join(keys).encode()))


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _speech_rms(y):
    """RMS over the louder half of 20 ms frames, so pauses don't skew the SNR."""
    n = len(y) // 320 * 320
    fr = np.sqrt(np.mean(y[:n].reshape(-1, 320).astype(np.float64) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.sort(fr)[len(fr) // 2:] ** 2)) + 1e-12)


def babble(source, n, rng):
    """Sum of the other four library speeches, each at equal level with a
    random start, looped to length n."""
    out = np.zeros(n)
    for s in SOURCES:
        if s == source:
            continue
        b = audio.load(LIB / s / "audio.wav").astype(np.float64)
        b /= _speech_rms(b.astype(np.float32))
        off = int(rng.integers(0, len(b)))
        idx = (np.arange(n) + off) % len(b)
        out += b[idx]
    return out


def degrade(y, source, cond, key):
    cond = cond.replace("+match", "")  # same degraded audio as the unmatched condition
    p = CONDITIONS[cond]
    y = y.astype(np.float64)
    rng = _rng(key, cond)
    if "noise" in p:
        nz = babble(source, len(y), rng) if p["noise"] == "babble" else rng.standard_normal(len(y))
        nz *= _speech_rms(y.astype(np.float32)) / (_rms(nz) * 10 ** (p["snr"] / 20))
        y = y + nz
    if "rt60" in p:
        from scipy.signal import fftconvolve
        L = int(p["rt60"] * SR)
        t = np.arange(L) / SR
        h = rng.standard_normal(L) * np.exp(-6.9 * t / p["rt60"]) * 0.25
        h[0] = 1.0
        wet = fftconvolve(y, h)[: len(y)]
        y = wet * (_rms(y) / _rms(wet))
    if p.get("telephone"):
        from scipy.signal import butter, resample_poly, sosfiltfilt
        z = resample_poly(y, 1, 2)
        z = sosfiltfilt(butter(6, [300, 3400], btype="band", fs=8000, output="sos"), z)
        y = resample_poly(z, 2, 1)[: len(y)]
    if "mp3" in p:
        with tempfile.TemporaryDirectory() as d:
            w, m = Path(d) / "a.wav", Path(d) / "a.mp3"
            audio.save(w, y.astype(np.float32))
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(w), "-b:a", f"{p['mp3']}k", str(m)],
                           check=True)
            z = audio.load(m).astype(np.float64)
        y = np.pad(z, (0, max(0, len(y) - len(z))))[: len(y)]
    if "tempo" in p:
        import parselmouth
        from parselmouth.praat import call
        snd = parselmouth.Sound(y, SR)
        y = call(snd, "Lengthen (overlap-add)", 60, 500, p["tempo"]).values[0]
    pk = np.max(np.abs(y))
    if pk > 0.99:
        y *= 0.99 / pk
    return y.astype(np.float32)


def run(job):
    clip, cond = job
    from speechlens.contrast import compare
    from speechlens.explain import load_rubric, score
    from speechlens.pipeline import analyze_cached
    src = LIB / clip["source"]
    text = (src / "transcript.txt").read_text()
    ref_y = audio.load(src / "audio.wav")
    clean_y = audio.load(audio.clip_path(DS / "clips", clip["id"]))
    y = clean_y if cond == "clean" else degrade(clean_y, clip["source"], cond, clip["id"])
    if cond.endswith("+match"):
        from speechlens.channel import match_reference
        ref_y, _ = match_reference(ref_y, y)
    ref = analyze_cached(ref_y, text)
    par = analyze_cached(y, text)
    res = compare(ref, par)
    sc = score(res["regions"], len(par.words), par.duration, load_rubric("declamation"))
    # aligner robustness: word onsets vs. the clean recording's own alignment
    f = CONDITIONS[cond.replace("+match", "")].get("tempo", 1.0)
    clean = analyze_cached(clean_y, text)
    errs = [abs(a.start - b.start * f) for a, b in zip(par.words, clean.words) if a.text == b.text]
    return {"id": clip["id"], "cond": cond, "kind": clip["kind"], "duration": par.duration,
            "labels": [dict(l, start=l["start"] * f, end=l["end"] * f) for l in clip["labels"]],
            "preds": [{"type": r.type, "start": r.start, "end": r.end, "severity": r.severity, "scope": r.scope}
                      for r in res["regions"]],
            "score": sc["overall"], "align_err_ms": float(np.median(errs) * 1000) if errs else None}


def summarize(rows):
    out = {}
    for cond in CONDITIONS:
        R = [r for r in rows if r["cond"] == cond]
        if not R:
            continue
        fl = [r for r in R if r["labels"]]
        nul = [r for r in R if not r["labels"]]
        tp = fp = fn = 0
        ious, by_type = [], {}
        for r in fl:
            loc = [p for p in r["preds"] if p["scope"] == "local"]
            m = match(r["labels"], loc, 0.3)
            tp += len(m)
            fp += len(loc) - len(m)
            fn += len(r["labels"]) - len(m)
            ious += [x for _, _, x in m]
            for i, l in enumerate(r["labels"]):
                e = by_type.setdefault(l["type"], [0, 0])
                e[1] += 1
                e[0] += any(a == i for a, _, _ in m)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        mins = sum(r["duration"] for r in nul) / 60
        out[cond] = {"label": LABEL[cond], "precision": round(p, 3), "recall": round(rc, 3),
                     "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0,
                     "mean_tiou": round(float(np.mean(ious)), 3) if ious else None,
                     "null_false_per_min": round(sum(len(r["preds"]) for r in nul) / mins, 3) if mins else None,
                     "null_local_false_per_min": round(sum(len([p for p in r["preds"] if p["scope"] == "local"])
                                                           for r in nul) / mins, 3) if mins else None,
                     "null_mean_score": round(float(np.mean([r["score"] for r in nul])), 1) if nul else None,
                     "align_err_ms": round(float(np.median([r["align_err_ms"] for r in R
                                                            if r["align_err_ms"] is not None])), 1),
                     "recall_by_type": {k: round(v[0] / v[1], 2) for k, v in sorted(by_type.items())},
                     "n_flawed": len(fl), "n_null": len(nul)}
    return out


def figure(summary, path):
    """F1 and false findings per condition; channel-matched variants drawn as a
    second, lighter bar next to their condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = [c for c in summary if not c.endswith("+match")]
    y = np.arange(len(base))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
    for ax, key, col, title, xmax in ((axes[0], "f1", "#2f6fdf", "F1 locating flaws (tIoU >= 0.3)", 1.0),
                                      (axes[1], "null_false_per_min", "#b8344a", "False findings / min, clean speech", None)):
        vals = [summary[c][key] or 0 for c in base]
        ax.barh(y + 0.18, vals, 0.36, color=col, label="as recorded")
        mv = [summary.get(c + "+match", {}).get(key) for c in base]
        ax.barh([yy - 0.18 for yy, v in zip(y, mv) if v is not None], [v for v in mv if v is not None], 0.36,
                color=col, alpha=0.4, label="reference channel-matched")
        for yy, v in zip(y, vals):
            ax.text(v + (xmax or max(vals)) * 0.01, yy + 0.18, f"{v:.2f}", va="center", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        if xmax:
            ax.set_xlim(0, xmax)
    axes[0].set_yticks(y, [summary[c]["label"] for c in base], fontsize=8)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(str(path).replace(".png", ".svg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="one speech only, for a smoke test")
    args = ap.parse_args()
    man = json.loads((DS / "manifest.json").read_text())
    clips = [c for c in man["clips"] if (c["kind"] == "single" and c["labels"][0]["severity"] == 4)
             or (c["kind"] == "mixed" and c.get("mix_level") == 6)
             or (c["kind"] == "control" and c.get("control") == "identity")]
    if args.quick:
        clips = [c for c in clips if c["source"] == "jfk"]
    jobs = [(c, cond) for cond in CONDITIONS for c in clips]
    os.environ.setdefault("SPEECHLENS_THREADS", str(max(1, (os.cpu_count() or 1) // args.workers)))
    part = OUT / "stress_rows.jsonl"
    done = {}
    if part.exists():
        for line in part.read_text().splitlines():
            r = json.loads(line)
            done[(r["id"], r["cond"])] = r
    todo = [j for j in jobs if (j[0]["id"], j[1]) not in done]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, part.open("a") as fh:
        for k, r in enumerate(ex.map(run, todo, chunksize=1)):
            done[(r["id"], r["cond"])] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if k % 10 == 0:
                print(f"{k + 1}/{len(todo)} {r['cond']} {r['id']}", flush=True)
    rows = [done[(c["id"], cond)] for c, cond in jobs]
    summary = summarize(rows)
    OUT.mkdir(exist_ok=True)
    (OUT / "stress.json").write_text(json.dumps({"conditions": summary, "n_clips": len(clips),
                                                 "flaw_types": FLAW_TYPES}, indent=1))
    figure(summary, ROOT / "docs/figures/stress.png")
    print(f"{'condition':18s}   P     R     F1   tIoU  null/min  align(ms)")
    for c, s in summary.items():
        print(f"{s['label']:18s} {s['precision']:.2f}  {s['recall']:.2f}  {s['f1']:.2f}  {s['mean_tiou'] or 0:.2f}  "
              f"{s['null_false_per_min']:7.2f}  {s['align_err_ms']}")


if __name__ == "__main__":
    main()
