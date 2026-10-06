"""Channel matching: put the reference through the participant's channel.

A contrastive measurement is only fair if both readings are heard through the
same channel. When the participant recorded in a noisier room than the
reference, background noise fills their pauses, lifts quiet word tails and adds
high-frequency energy - all of which the contrast would otherwise report as
delivery flaws (shorter pauses, longer words, "clearer" articulation).

We estimate the participant's noise from their quietest frames, and if it
exceeds the reference's noise floor, add noise with exactly the missing
spectrum to the reference (seeded, so the result is reproducible). Both
readings then carry the same noise, and its effect cancels in the comparison.
Clean participant recordings leave the reference untouched.
"""
from __future__ import annotations

import zlib

import numpy as np

from .audio import SR

N_FFT, HOP = 512, 256
MIN_EXCESS_DB = 3.0       # match only when the participant is clearly noisier


def _stft(y):
    import librosa
    return librosa.stft(y.astype(np.float32), n_fft=N_FFT, hop_length=HOP)


def noise_psd(y: np.ndarray, quantile: float = 0.15) -> tuple[np.ndarray, float]:
    """Mean power spectrum of the quietest `quantile` of frames, and its level
    in dB relative to the recording's speech level (95th pct frame power)."""
    S = np.abs(_stft(y)) ** 2
    e = S.sum(axis=0)
    k = max(3, int(quantile * S.shape[1]))
    quiet = np.argsort(e)[:k]
    psd = S[:, quiet].mean(axis=1)
    speech = np.percentile(e, 95) + 1e-12
    return psd, float(10 * np.log10(psd.sum() / speech + 1e-12))


def match_reference(ref_y: np.ndarray, par_y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return the reference with the participant's excess noise added, and a
    description. Levels are taken relative to each recording's own speech
    level, so recording gain does not matter."""
    p_psd, p_db = noise_psd(par_y)
    r_psd, r_db = noise_psd(ref_y)
    info = {"participant_noise_db": round(p_db, 1), "reference_noise_db": round(r_db, 1), "matched": False}
    if p_db - r_db < MIN_EXCESS_DB:
        return ref_y, info
    # scale both noise spectra to the reference's speech level, take the excess
    S_ref = np.abs(_stft(ref_y)) ** 2
    r_speech = np.percentile(S_ref.sum(axis=0), 95)
    S_par = np.abs(_stft(par_y)) ** 2
    p_speech = np.percentile(S_par.sum(axis=0), 95)
    excess = np.maximum(p_psd * (r_speech / p_speech) - r_psd, 0.0)
    rng = np.random.default_rng(zlib.crc32(par_y[: SR * 5].tobytes()))
    n_frames = S_ref.shape[1]
    phase = np.exp(2j * np.pi * rng.random((len(excess), n_frames)))
    # per-frame magnitude for a noise of power `excess` under librosa's STFT scaling
    mag = np.sqrt(excess)[:, None] * np.ones((1, n_frames))
    import librosa
    noise = librosa.istft(mag * phase, hop_length=HOP, length=len(ref_y))
    # calibrate: the synthesized noise's measured PSD should equal `excess`
    got = (np.abs(_stft(noise)) ** 2).mean(axis=1).sum() + 1e-12
    noise *= np.sqrt(excess.sum() / got)
    out = (ref_y + noise).astype(np.float32)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out *= 0.99 / pk
    info.update(matched=True, added_noise_db=round(float(10 * np.log10(excess.sum() / r_speech + 1e-12)), 1))
    return out, info
