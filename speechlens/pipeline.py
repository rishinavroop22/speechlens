"""End-to-end evaluation: reference + participant -> report (JSON-ready)."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from . import audio, features
from .contrast import compare
from .explain import explain, load_rubric, score, DIM_LABEL

CACHE = Path(__file__).resolve().parent.parent / ".cache"


def _key(y: np.ndarray, text: str) -> str:
    h = hashlib.sha1(y.tobytes())
    h.update(text.encode())
    h.update(b"v4")
    return h.hexdigest()[:16]


def analyze_cached(y: np.ndarray, text: str) -> features.Analysis:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{_key(y, text)}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    an = features.analyze(y, text)
    p.write_bytes(pickle.dumps(an))
    return an


def warp_reference(ref: features.Analysis, par: features.Analysis, pairs) -> dict:
    """Map the reference contours onto the participant's timeline using the
    paired word boundaries as anchors, so the overlay compares the same words
    at the same x position even when the participant rushed or paused."""
    pa, ra = [0.0], [0.0]
    for i, j in pairs:
        for rt, pt in ((ref.words[i].start, par.words[j].start), (ref.words[i].end, par.words[j].end)):
            if pt > pa[-1] and rt > ra[-1]:
                pa.append(pt)
                ra.append(rt)
    pa.append(par.duration)
    ra.append(max(ra[-1] + 1e-3, ref.duration))
    t = par.ff.t
    rt = np.interp(t, pa, ra)
    idx = np.clip((rt / features.HOP).astype(int), 0, len(ref.ff.t) - 1)
    return {"f0_st": ref.ff.f0_st[idx], "intensity_db": ref.ff.intensity_db[idx],
            "hiband": ref.ff.hiband[idx], "ref_time": rt}


def _series(x, step):
    return [None if not np.isfinite(v) else round(float(v), 2) for v in np.asarray(x)[::step]]


def evaluate(ref_y, ref_text, par_y, par_text=None, rubric: str = "declamation", step: int = 2) -> dict:
    par_text = par_text or ref_text
    ref = analyze_cached(ref_y, ref_text)
    par = analyze_cached(par_y, par_text)
    from .contrast import pair_words

    pairs = pair_words(ref, par)
    res = compare(ref, par)
    for r in res["regions"]:
        r.explanation = explain(r)
    sc = score(res["regions"], len(par.words), par.duration, load_rubric(rubric))
    warped = warp_reference(ref, par, pairs)
    return {
        "duration": round(par.duration, 3),
        "reference_duration": round(ref.duration, 3),
        "align_method": par.align_method,
        "score": sc,
        "dimension_labels": DIM_LABEL,
        "regions": [r.to_dict() for r in res["regions"]],
        "global": res["global"],
        "coverage": res["coverage"],
        "words": [w.to_dict() for w in par.words],
        "ref_words": [w.to_dict() for w in ref.words],
        "series": {
            "t": _series(par.ff.t, step),
            "participant": {"f0_st": _series(par.ff.f0_st, step), "intensity_db": _series(par.ff.intensity_db, step),
                            "hiband": _series(par.ff.hiband * 100, step)},
            "reference": {"f0_st": _series(warped["f0_st"], step), "intensity_db": _series(warped["intensity_db"], step),
                          "hiband": _series(warped["hiband"] * 100, step)},
            "f0_median_hz": {"participant": round(par.ff.f0_median_hz, 1), "reference": round(ref.ff.f0_median_hz, 1)},
        },
    }


def evaluate_files(ref_audio, ref_txt, par_audio, par_txt=None, rubric="declamation"):
    rt = Path(ref_txt).read_text()
    pt = Path(par_txt).read_text() if par_txt else None
    return evaluate(audio.load(ref_audio), rt, audio.load(par_audio), pt, rubric)
