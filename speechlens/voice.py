"""Voice-quality profile of a reading (descriptive, not used for flaw detection).

Standard clinical/phonetic measures via Praat, computed over the voiced parts
of the whole reading, for the participant and the reference side by side:

  jitter (local, %)   cycle-to-cycle variation of the pitch period: roughness
  shimmer (local, %)  cycle-to-cycle variation of amplitude: roughness/breathiness
  HNR (dB)            harmonics-to-noise ratio: breathiness/hoarseness
  CPPS (dB)           smoothed cepstral peak prominence: overall voice clarity,
                      the most robust single measure of dysphonia
  F0 median, F0 variability (st), voiced share of speech time

Reference values for typical adult voices (Praat defaults / clinical
literature): jitter < 1.04 %, shimmer < 3.81 %, HNR > 20 dB. Phone microphones
and room noise raise jitter/shimmer and lower HNR/CPPS, so the comparison with
the reference, not the absolute value, is what the dashboard shows.
"""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from .audio import SR

NORMS = {"jitter_pct": (1.04, "max"), "shimmer_pct": (3.81, "max"), "hnr_db": (20.0, "min")}


def profile(y: np.ndarray, f0_median_hz: float | None = None) -> dict:
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    med = f0_median_hz or 150.0
    floor, ceil = max(50.0, med * 0.55), min(800.0, med * 2.4)
    out = {}
    try:
        pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceil)
        f = pitch.selected_array["frequency"]
        v = f[f > 0]
        out["f0_median_hz"] = round(float(np.median(v)), 1) if len(v) else None
        out["f0_sd_st"] = round(float(np.std(12 * np.log2(v / np.median(v)))), 2) if len(v) > 10 else None
        out["voiced_share"] = round(float(len(v) / max(1, len(f))), 3)
        pp = call([snd, pitch], "To PointProcess (cc)")
        out["jitter_pct"] = round(100 * call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 2)
        out["shimmer_pct"] = round(100 * call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 2)
    except Exception:  # noqa: BLE001 - silence or unvoiced input
        pass
    try:
        hnr = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=floor)
        hv = hnr.values[0]
        hv = hv[hv > -100]
        out["hnr_db"] = round(float(np.mean(hv)), 1) if len(hv) else None
    except Exception:  # noqa: BLE001
        pass
    try:
        cep = call(snd, "To PowerCepstrogram", floor, 0.002, 5000.0, 50.0)
        out["cpps_db"] = round(call(cep, "Get CPPS", "no", 0.02, 0.0005, floor, ceil, 0.05, "parabolic",
                                    0.001, 0.0, "Straight", "Robust"), 2)
    except Exception:  # noqa: BLE001 - signature differs across Praat versions
        try:
            cep = call(snd, "To PowerCepstrogram", floor, 0.002, 5000.0, 50.0)
            out["cpps_db"] = round(call(cep, "Get CPPS", "no", 0.02, 0.0005, floor, ceil, 0.05, "Parabolic",
                                        0.001, 0.0, "Exponential decay", "Robust slow"), 2)
        except Exception:  # noqa: BLE001
            out["cpps_db"] = None
    return out


def compare_profiles(ref: dict, par: dict) -> list[dict]:
    """Rows for the dashboard: measure, reference, participant, plain reading."""
    rows = []
    spec = [("jitter_pct", "Jitter (local)", "%", "higher = rougher"),
            ("shimmer_pct", "Shimmer (local)", "%", "higher = rougher or breathier"),
            ("hnr_db", "Harmonics-to-noise ratio", "dB", "lower = breathier or hoarser"),
            ("cpps_db", "Cepstral peak prominence (CPPS)", "dB", "lower = less clear voice"),
            ("f0_median_hz", "Median pitch", "Hz", "speaker's own register"),
            ("f0_sd_st", "Pitch variability", "st", "lower = flatter delivery"),
            ("voiced_share", "Voiced share of time", "", "lower = more pauses or unvoiced speech")]
    for key, label, unit, note in spec:
        r, p = ref.get(key), par.get(key)
        if r is None and p is None:
            continue
        flag = None
        if key in NORMS and p is not None:
            lim, kind = NORMS[key]
            flag = "outside typical range" if (p > lim if kind == "max" else p < lim) else "typical"
        rows.append({"key": key, "label": label, "unit": unit, "reference": r, "participant": p, "note": note,
                     "norm": flag})
    return rows
