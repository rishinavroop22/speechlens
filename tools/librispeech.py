"""Generalisation to unseen speakers: LibriSpeech dev-clean (CC BY 4.0).

All five gold speakers are male US presidents. This test builds new gold
passages from LibriSpeech read speech (audiobook readers, balanced by sex),
injects flaws with the same engine, and evaluates with the thresholds,
gates and severity tables frozen as fitted on our dataset - nothing is tuned
on these speakers.

Per speaker: consecutive utterances of one chapter are joined (with 0.35 s of
the reader's own room tone between them) into a passage of about 60 s; then
one clip per flaw type at severity 4, plus the unflawed passage as a null.

Usage:
  python tools/librispeech.py --root path/to/LibriSpeech --speakers 20
Writes data/librispeech/ (audio, labels) and results/librispeech.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio  # noqa: E402
from speechlens.inject import ALL_TYPES, Injector  # noqa: E402
from tools.build_dataset import _seed  # noqa: E402
from tools.evaluate import match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/librispeech"
SEV = 4


def speakers_meta(root: Path) -> dict:
    meta = {}
    f = root / "SPEAKERS.TXT"
    if f.exists():
        for line in f.read_text(errors="ignore").splitlines():
            if line.startswith(";") or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                meta[parts[0]] = {"sex": parts[1], "subset": parts[2]}
    return meta


def passage(chapter_dir: Path, target_s: float = 60.0):
    trans = next(chapter_dir.glob("*.trans.txt"), None)
    if trans is None:
        return None
    lines = [ln.split(" ", 1) for ln in trans.read_text().splitlines() if " " in ln]
    ys, words = [], []
    for uid, text in lines:
        f = chapter_dir / f"{uid}.flac"
        if not f.exists():
            continue
        y = audio.load(f)
        ys.append(y)
        words.append(text.strip().lower())
        if sum(len(v) for v in ys) / audio.SR >= target_s:
            break
    if not ys:
        return None
    # room tone: the quietest 50 ms of the first utterance, repeated
    q = ys[0][: audio.SR // 2]
    k = len(q) // 800 * 800
    seg = q[:k].reshape(-1, 800)
    tone = seg[np.argmin((seg ** 2).mean(axis=1))]
    gap = np.tile(tone, int(0.35 * audio.SR / 800) + 1)[: int(0.35 * audio.SR)]
    out = []
    for i, y in enumerate(ys):
        out.append(y)
        if i < len(ys) - 1:
            out.append(gap)
    text = ". ".join(w.capitalize() for w in words) + "."
    return audio.peak_normalize(np.concatenate(out)), re.sub(r"\s+", " ", text)


def run(job):
    cid, y_path, text, ref_path = job
    from speechlens.contrast import compare
    from speechlens.pipeline import analyze_cached, prepare_reference
    ref_y = audio.load(ref_path)
    y = audio.load(y_path)
    ref_y, _ = prepare_reference(ref_y, y)
    ref, par = analyze_cached(ref_y, text), analyze_cached(y, text)
    res = compare(ref, par, y)
    return {"id": cid, "preds": [{"type": r.type, "start": r.start, "end": r.end, "severity": r.severity,
                                  "scope": r.scope} for r in res["regions"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing dev-clean/ (or a LibriSpeech folder)")
    ap.add_argument("--speakers", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.root)
    sub = root / "dev-clean" if (root / "dev-clean").exists() else root
    meta = speakers_meta(root if (root / "SPEAKERS.TXT").exists() else root.parent)
    spk_dirs = sorted(p for p in sub.iterdir() if p.is_dir())
    # balance by sex when the metadata is there
    f = [p for p in spk_dirs if meta.get(p.name, {}).get("sex") == "F"]
    m = [p for p in spk_dirs if meta.get(p.name, {}).get("sex") == "M"]
    chosen = (f[: a.speakers // 2] + m[: a.speakers // 2]) if f and m else spk_dirs[: a.speakers]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips, jobs = [], []
    from speechlens import features
    for sd in chosen:
        ch = sorted(p for p in sd.iterdir() if p.is_dir())
        if not ch:
            continue
        got = passage(ch[0])
        if got is None:
            continue
        y, text = got
        sid = f"ls{sd.name}"
        gold = OUT_DIR / f"{sid}__gold.wav"
        audio.save(gold, y)
        an = features.analyze(y, text)
        clips.append({"id": f"{sid}__gold", "speaker": sd.name, "sex": meta.get(sd.name, {}).get("sex"),
                      "type": None, "labels": [], "duration": round(len(y) / audio.SR, 2)})
        for t in ALL_TYPES:
            inj = Injector(y, an.words, an.ff.f0_median_hz, seed=_seed(sid, t, SEV, 0))
            out, nw, lb = inj.apply([(t, SEV)])
            if not lb:
                continue
            cid = f"{sid}__{t}_s{SEV}"
            audio.save(OUT_DIR / f"{cid}.wav", out)
            clips.append({"id": cid, "speaker": sd.name, "sex": meta.get(sd.name, {}).get("sex"), "type": t,
                          "labels": [l.to_dict() for l in lb], "duration": round(len(out) / audio.SR, 2)})
            jobs.append((cid, str(OUT_DIR / f"{cid}.wav"), text, str(gold)))
        print(f"  {sid} ({meta.get(sd.name, {}).get('sex', '?')}): {len(y) / audio.SR:.0f} s, {len(an.words)} words",
              flush=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(clips, indent=1))
    os.environ.setdefault("SPEECHLENS_THREADS", str(max(1, (os.cpu_count() or 1) // a.workers)))
    with ProcessPoolExecutor(a.workers) as ex:
        preds = {r["id"]: r for r in ex.map(run, jobs, chunksize=1)}

    def block(cs):
        tp = fp = fn = 0
        for c in cs:
            loc = [p for p in preds[c["id"]]["preds"] if p["scope"] == "local"]
            mm = match(c["labels"], loc, 0.3)
            tp, fp, fn = tp + len(mm), fp + len(loc) - len(mm), fn + len(c["labels"]) - len(mm)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
                "tp": tp, "fp": fp, "fn": fn}
    flawed = [c for c in clips if c["labels"]]
    res = {"speakers": len({c["speaker"] for c in clips}), "severity": SEV, "all": block(flawed),
           "by_sex": {s: block([c for c in flawed if c["sex"] == s]) for s in ("F", "M")},
           "by_type": {t: block([c for c in flawed if c["type"] == t]) for t in ALL_TYPES}}
    (ROOT / "results/librispeech.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
