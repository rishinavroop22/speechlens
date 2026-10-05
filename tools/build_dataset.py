"""Build the contrastive dataset.

  data/raw/<id>.<mp3|wav|ogg|m4a>  + data/raw/<id>.txt   (full speech + transcript)
  data/raw/sources.json  (title, speaker, year, url, licence per id; optional)
  data/raw/excerpts.json (optional: {"<id>": "first words ... last words"})
        |
        v
  data/library/<id>/{audio.wav, transcript.txt, meta.json}   gold excerpts
  data/dataset/clips/<clip>.wav                               spectrum clips
  data/dataset/labels/<clip>.json                             labels + word timings
  data/dataset/manifest.json                                  index of everything

Usage:  python tools/build_dataset.py [--excerpt-sec 75] [--seeds 1]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio, align, features  # noqa: E402
from speechlens.inject import Injector, FLAW_TYPES  # noqa: E402
from speechlens.text import tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, LIB, DS = ROOT / "data/raw", ROOT / "data/library", ROOT / "data/dataset"
AUDIO_EXT = (".wav", ".mp3", ".ogg", ".oga", ".flac", ".m4a", ".webm", ".opus", ".aac")


def _seed(*parts) -> int:
    """Process-independent seed (Python's hash() is salted per run)."""
    return zlib.crc32("|".join(map(str, parts)).encode())


def greedy_words(lp: np.ndarray) -> list[tuple[str, float, float]]:
    """Greedy CTC decode with word timestamps (used only to locate excerpts)."""
    _, vocab = align._session()
    inv = {v: k for k, v in vocab.items()}
    ids = lp.argmax(axis=1)
    words, cur, t0, prev = [], "", None, -1
    for t, i in enumerate(ids):
        if i != prev and i != 0:
            ch = inv.get(int(i), "")
            if ch == "|":
                if cur:
                    words.append((cur, t0 * align.FRAME, t * align.FRAME))
                cur, t0 = "", None
            elif ch not in ("<s>", "</s>", "<unk>"):
                if t0 is None:
                    t0 = t
                cur += ch
        prev = i
    if cur:
        words.append((cur, t0 * align.FRAME, len(ids) * align.FRAME))
    return words


def locate(y, transcript: str, seconds: float):
    """Find where the speech text starts in the recording (skipping oath,
    applause, introductions) and cut at the sentence end nearest `seconds`
    later. Uses a greedy CTC transcript only to anchor matched words."""
    lp = align.emissions(y[: int(10 * 60 * audio.SR)])  # excerpts sit near the start
    hyp = greedy_words(lp)
    toks = transcript.split()[:600]
    want, keep = [], []  # normalized words + index of the source token
    for i, tk in enumerate(toks):
        for t in tokenize(tk):
            want.append(t.norm.replace("|", ""))
            keep.append(i)
    got = [w for w, _, _ in hyp]
    sm = difflib.SequenceMatcher(a=want, b=got, autojunk=False)
    t_of = {}
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 3]
    # the speech starts at the first run of >= 5 consecutive matching words;
    # shorter runs before it are introductions / applause noise
    start_a = min((b.a for b in blocks if b.size >= 5), default=0)
    for b in blocks:
        if b.a >= start_a:
            for k in range(b.size):
                t_of[b.a + k] = (hyp[b.b + k][1], hyp[b.b + k][2])
    if len(t_of) < 10:
        raise RuntimeError("could not find the transcript in the audio")
    first = min(t_of)
    t0 = t_of[first][0]
    # back up to the start of the sentence the first anchor sits in
    i0 = keep[first]
    back = 0
    while i0 - back - 1 >= 0 and back < 30 and not re.search(r"[.!?:][\"'”’)]*$", toks[i0 - back - 1]):
        back += 1
    if back < 30:
        t0 = max(0.0, t0 - 1.0 * back - 1.0)
        i0 -= back
    else:
        back = 0
    # sentence ends with a matched timestamp
    ends = [k for k in t_of if k > first + 20 and re.search(r"[.!?][\"'”’)]*$", toks[keep[k]])]
    # closest sentence end to the target length, never shorter than 70 % of it when possible
    long_enough = [k for k in ends if t_of[k][1] - t0 >= 0.7 * seconds] or ends
    best = min(long_enough, key=lambda k: abs((t_of[k][1] - t0) - seconds))
    text = " ".join(toks[i0: keep[best] + 1])
    return max(0.0, t0 - 0.5), t_of[best][1] + 0.7, text, len(t_of)


def build_gold(seconds: float, only=None):
    meta_all = json.loads((RAW / "sources.json").read_text()) if (RAW / "sources.json").exists() else {}
    ex_all = json.loads((RAW / "excerpts.json").read_text()) if (RAW / "excerpts.json").exists() else {}
    golds = []
    for txt in sorted(RAW.glob("*.txt")):
        sid = txt.stem
        if only and sid not in only:
            if (LIB / sid / "audio.wav").exists():  # reuse previous build
                y = audio.load(LIB / sid / "audio.wav")
                golds.append((sid, y, features.analyze(y, (LIB / sid / "transcript.txt").read_text())))
            continue
        src = next((p for p in RAW.glob(sid + ".*") if p.suffix.lower() in AUDIO_EXT), None)
        if src is None:
            print(f"  ! {sid}: transcript without audio, skipped")
            continue
        y = audio.peak_normalize(audio.load(src))
        full = re.sub(r"\s+", " ", txt.read_text()).strip()
        excerpt = ex_all.get(sid) or full
        if align.model_available() and len(y) / audio.SR > seconds * 1.3:
            a, b, excerpt, nm = locate(y, excerpt, seconds)
            y = audio.crop(y, a, b)
            print(f"  {sid}: excerpt {a:.1f}-{b:.1f}s ({nm} anchor words)")
        an = features.analyze(y, excerpt)
        conf = float(np.mean([w.conf for w in an.words])) if an.words else 0
        d = LIB / sid
        d.mkdir(parents=True, exist_ok=True)
        audio.save(d / "audio.wav", y)
        (d / "transcript.txt").write_text(excerpt + "\n")
        meta = {"title": sid, "speaker": sid, "year": None, "source": None, "licence": "public domain"} | meta_all.get(sid, {})
        meta |= {"duration": round(len(y) / audio.SR, 2), "words": len(an.words), "align_conf": round(conf, 3),
                 "f0_median_hz": round(an.ff.f0_median_hz, 1), "align_method": an.align_method}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        (d / "alignment.json").write_text(json.dumps([w.to_dict() for w in an.words], indent=1))
        golds.append((sid, y, an))
        print(f"  {sid}: {meta['duration']}s, {len(an.words)} words, mean align conf {conf:.2f}")
    return golds


def build_spectrum(golds, seeds: int):
    (DS / "clips").mkdir(parents=True, exist_ok=True)
    (DS / "labels").mkdir(parents=True, exist_ok=True)
    clips = []

    def emit(cid, sid, y, words, labels, kind, extra=None):
        audio.save(DS / "clips" / f"{cid}.wav", y)
        rec = {"id": cid, "source": sid, "kind": kind, "duration": round(len(y) / audio.SR, 2),
               "labels": [l.to_dict() if hasattr(l, "to_dict") else l for l in labels],
               "max_severity": max([l.severity if hasattr(l, "severity") else l["severity"] for l in labels], default=0)}
        rec |= extra or {}
        (DS / "labels" / f"{cid}.json").write_text(json.dumps(rec | {"words": [w.to_dict() for w in words]}, indent=1))
        clips.append(rec)

    for sid, y, an in golds:
        f0 = an.ff.f0_median_hz
        emit(f"{sid}__gold", sid, y, an.words, [], "control", {"control": "identity"})
        # speaker-invariance controls: same delivery, different voice/gain
        for name, semis, gain in (("pitch_up4", 4, 0), ("pitch_down4", -4, 0), ("quiet12db", 0, -12)):
            import parselmouth
            from parselmouth.praat import call
            snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=audio.SR)
            if semis:
                m = call(snd, "To Manipulation", 0.01, max(50, f0 * 0.55), min(800, f0 * 2.4))
                pt = call(m, "Extract pitch tier")
                call(pt, "Multiply frequencies", 0, snd.duration, 2 ** (semis / 12))
                call([m, pt], "Replace pitch tier")
                yy = call(m, "Get resynthesis (overlap-add)").values[0].astype(np.float32)[: len(y)]
            else:
                yy = y * 10 ** (gain / 20)
            emit(f"{sid}__ctrl_{name}", sid, yy, an.words, [], "control", {"control": name})
        for seed in range(seeds):
            for ft in FLAW_TYPES:
                for sev in range(1, 6):
                    inj = Injector(y, an.words, f0, seed=_seed(sid, ft, sev, seed))
                    out, nw, lb = inj.apply([(ft, sev)])
                    emit(f"{sid}__{ft}_s{sev}" + (f"_r{seed}" if seed else ""), sid, out, nw, lb, "single")
            # mixed clips: the near-perfect -> botched gradient
            rng = np.random.default_rng(_seed(sid, "mix", seed))
            for level, (lo, hi, nf) in enumerate([(1, 2, 2), (1, 3, 2), (2, 3, 3), (2, 4, 3), (3, 4, 4),
                                                  (3, 5, 4), (4, 5, 5), (5, 5, 6)], start=1):
                types = rng.choice(FLAW_TYPES, size=nf, replace=False)
                specs = [(str(t), int(rng.integers(lo, hi + 1))) for t in types]
                inj = Injector(y, an.words, f0, seed=int(rng.integers(2**31)))
                out, nw, lb = inj.apply(specs)
                emit(f"{sid}__mix{level}" + (f"_r{seed}" if seed else ""), sid, out, nw, lb, "mixed", {"mix_level": level})
        print(f"  {sid}: {sum(c['source'] == sid for c in clips)} clips")
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excerpt-sec", type=float, default=75)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="rebuild only these speech ids (others are kept)")
    args = ap.parse_args()
    print("aligner:", "wav2vec2-ctc" if align.model_available() else "FALLBACK (model missing)")
    print("gold excerpts:")
    golds = build_gold(args.excerpt_sec, args.only)
    print("spectrum:")
    clips = build_spectrum(golds, args.seeds)
    human = json.loads((DS / "human.json").read_text()) if (DS / "human.json").exists() else []
    manifest = {"version": 1, "flaw_types": FLAW_TYPES, "severity_scale": "1 = near-perfect ... 5 = egregious",
                "sources": [s for s, _, _ in golds], "clips": clips + human}
    (DS / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {len(golds)} gold speeches, {len(clips)} synthetic clips, {len(human)} human clips")


if __name__ == "__main__":
    main()
