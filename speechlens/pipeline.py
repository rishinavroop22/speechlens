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


def recording_check(ref_y, par_y) -> dict:
    """Is the comparison fair? Contrast assumes both readings were heard through
    similar channels. Background noise in the participant's pauses, measured
    relative to their own speech level, beyond the reference's makes timing and
    clarity findings unreliable (tools/stress.py: at babble SNR 20 dB, about
    9 dB of excess noise, precision falls from 0.90 to 0.30)."""
    from .channel import noise_psd
    _, p_db = noise_psd(par_y)
    _, r_db = noise_psd(ref_y)
    excess = p_db - r_db
    warn = None
    if excess > 6:
        warn = (f"The participant recording is {excess:.0f} dB noisier than the reference. Background noise fills "
                "pauses and word edges, so timing and clarity findings may reflect the room rather than the "
                "delivery. The reference was given matching background noise to compensate, which helps only "
                "partly; re-record somewhere quieter for a reliable result.")
    return {"participant_noise_db": round(p_db, 1), "reference_noise_db": round(r_db, 1),
            "excess_db": round(excess, 1), "warning": warn}


def prepare_reference(ref_y, par_y):
    """The reference as it is compared: when the participant recording is much
    noisier, the reference is put through the participant's noise channel so
    the noise cancels in the contrast (a partial mitigation, tools/stress.py)."""
    check = recording_check(ref_y, par_y)
    if check["warning"]:
        from .channel import match_reference
        ref_y, info = match_reference(ref_y, par_y)
        check["channel_matched"] = info["matched"]
    return ref_y, check


def _series(x, step):
    return [None if not np.isfinite(v) else round(float(v), 2) for v in np.asarray(x)[::step]]


def evaluate(ref_y, ref_text, par_y, par_text=None, rubric: str = "declamation", step: int = 2) -> dict:
    par_text = par_text or ref_text
    ref_y, check = prepare_reference(ref_y, par_y)
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
        "recording_check": check,
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
