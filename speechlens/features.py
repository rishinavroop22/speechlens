"""Acoustic feature extraction on a shared 10 ms grid, then aggregation to
words and inter-word gaps.

Speaker-agnostic by construction:
  * F0 is tracked with a two-pass, speaker-adaptive range (De Looze & Hirst)
    and expressed in semitones relative to the speaker's own median F0;
  * intensity is in dB relative to the speaker's median speech level;
so a deep male voice and a high female voice reading identically produce the
same normalized contours, and recording gain does not matter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import parselmouth

from .align import Word
from .audio import SR
from .text import count_syllables

HOP = 0.01


@dataclass
class FrameFeatures:
    t: np.ndarray            # seconds
    f0_hz: np.ndarray        # nan when unvoiced
    f0_st: np.ndarray        # semitones re speaker median (nan unvoiced)
    intensity_db: np.ndarray  # dB re speaker median speech level
    hnr_db: np.ndarray       # harmonics-to-noise ratio (nan unvoiced)
    centroid_hz: np.ndarray
    hiband: np.ndarray       # share of energy above 2 kHz (articulation)
    hiband4: np.ndarray      # share of energy above 4 kHz (fricatives, bursts)
    flux: np.ndarray         # spectral flux (onset/articulation activity)
    mfcc: np.ndarray         # [13, n]
    f0_median_hz: float
    level_median_db: float

    def at(self, a: float, b: float) -> slice:
        return slice(int(a / HOP), max(int(a / HOP) + 1, int(b / HOP)))

    def to_json(self, step: int = 2) -> dict:
        def c(x):
            return [None if not np.isfinite(v) else round(float(v), 2) for v in x[::step]]
        return {"t": c(self.t), "f0_st": c(self.f0_st), "intensity_db": c(self.intensity_db),
                "hnr_db": c(self.hnr_db), "hiband": c(self.hiband * 100),
                "f0_median_hz": round(self.f0_median_hz, 1)}


def _resample_to(grid_t, src_t, src_v):
    out = np.interp(grid_t, src_t, np.nan_to_num(src_v, nan=0.0), left=0.0, right=0.0)
    mask = np.interp(grid_t, src_t, np.isfinite(src_v).astype(float), left=0, right=0) > 0.5
    out[~mask] = np.nan
    return out


def frame_features(y: np.ndarray) -> FrameFeatures:
    import librosa

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    n = int(np.ceil(len(y) / SR / HOP))
    t = np.arange(n) * HOP

    # --- F0, speaker-adaptive two-pass range
    p1 = snd.to_pitch_ac(time_step=HOP, pitch_floor=60, pitch_ceiling=700)
    f = p1.selected_array["frequency"]
    f = f[f > 0]
    if len(f) > 20:
        floor, ceil = max(50, np.percentile(f, 25) * 0.75), min(800, np.percentile(f, 75) * 1.5)
    else:
        floor, ceil = 75, 500
    p2 = snd.to_pitch_ac(time_step=HOP, pitch_floor=floor, pitch_ceiling=ceil,
                         voicing_threshold=0.45, octave_jump_cost=0.5)
    fh = p2.selected_array["frequency"].astype(float)
    fh[fh <= 0] = np.nan
    f0 = _resample_to(t, p2.xs(), fh)
    med = float(np.nanmedian(f0)) if np.isfinite(f0).any() else 150.0
    f0_st = 12 * np.log2(f0 / med)

    # --- intensity relative to the speaker's speech level
    it = snd.to_intensity(minimum_pitch=max(75, floor), time_step=HOP)
    ival = it.values[0]
    inten = np.interp(t, it.xs(), ival, left=ival[0], right=ival[-1])
    speech = inten > (np.percentile(inten, 95) - 30)
    level = float(np.median(inten[speech])) if speech.any() else float(np.median(inten))
    inten_rel = np.maximum(inten - level, -40.0)

    # --- voice quality
    hn = snd.to_harmonicity_cc(time_step=HOP, minimum_pitch=max(75, floor))
    hv = hn.values[0].astype(float)
    hv[hv < -50] = np.nan
    hnr = _resample_to(t, hn.xs(), hv)

    # --- spectral articulation measures
    hop_s = int(HOP * SR)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=hop_s, center=True)) ** 2
    S = S[:, :n] if S.shape[1] >= n else np.pad(S, ((0, 0), (0, n - S.shape[1])))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    tot = S.sum(axis=0) + 1e-12
    centroid = (freqs[:, None] * S).sum(axis=0) / tot
    hiband = S[freqs >= 2000].sum(axis=0) / tot
    hiband4 = S[freqs >= 4000].sum(axis=0) / tot
    logS = np.log(S + 1e-10)
    flux = np.r_[0, np.sqrt((np.diff(logS, axis=1).clip(min=0) ** 2).mean(axis=0))]
    mf = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=SR, n_mels=40)),
                              n_mfcc=13)
    silent = ~speech
    centroid[silent] = np.nan
    hiband[silent] = np.nan
    hiband4[silent] = np.nan
    return FrameFeatures(t, f0, f0_st, inten_rel, hnr, centroid, hiband, hiband4, flux, mf, med, level)


@dataclass
class WordFeatures:
    i: int
    text: str
    start: float
    end: float
    dur: float
    syllables: int
    rate_sps: float          # syllables per second within the word
    pause_after: float       # silence until next word (s)
    filled_after: float      # voiced sound inside that gap (s): "um", "uh"
    f0_mean_st: float
    f0_range_st: float
    int_mean_db: float
    int_peak_db: float
    hnr_db: float
    hiband: float
    hiband4: float
    conf: float
    punct_after: str | None
    extra: dict = field(default_factory=dict)


def _nanstat(fn, x, default=np.nan):
    x = x[np.isfinite(x)]
    return float(fn(x)) if len(x) else default


def word_features(words: list[Word], ff: FrameFeatures) -> list[WordFeatures]:
    out = []
    for k, w in enumerate(words):
        sl = ff.at(w.start, w.end)
        f0 = ff.f0_st[sl]
        syl = count_syllables(w.norm)
        dur = max(w.end - w.start, 0.02)
        nxt = words[k + 1].start if k + 1 < len(words) else w.end
        gap = max(0.0, nxt - w.end)
        filled = 0.0
        if gap > 0.15:
            g = ff.at(w.end + 0.03, nxt - 0.03)
            voiced = np.isfinite(ff.f0_hz[g]) & (ff.intensity_db[g] > -12)
            filled = float(voiced.sum() * HOP) if voiced.sum() * HOP > 0.12 else 0.0
        out.append(WordFeatures(
            i=w.i, text=w.text, start=w.start, end=w.end, dur=round(dur, 3), syllables=syl,
            rate_sps=round(syl / dur, 3), pause_after=round(gap, 3), filled_after=round(filled, 3),
            f0_mean_st=_nanstat(np.mean, f0), f0_range_st=_nanstat(lambda x: np.percentile(x, 90) - np.percentile(x, 10), f0, 0.0),
            int_mean_db=_nanstat(np.mean, ff.intensity_db[sl], -40), int_peak_db=_nanstat(np.max, ff.intensity_db[sl], -40),
            hnr_db=_nanstat(np.mean, ff.hnr_db[sl]), hiband=_nanstat(np.mean, ff.hiband[sl]), hiband4=_nanstat(np.mean, ff.hiband4[sl]),
            conf=w.conf, punct_after=w.punct_after))
    return out


@dataclass
class Analysis:
    """Everything known about one recording."""
    duration: float
    words: list[Word]
    wf: list[WordFeatures]
    ff: FrameFeatures
    align_method: str


def analyze(y: np.ndarray, transcript: str) -> Analysis:
    from .align import align

    words, method = align(y, transcript)
    ff = frame_features(y)
    return Analysis(len(y) / SR, words, word_features(words, ff), ff, method)
