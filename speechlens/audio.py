"""Audio I/O: every file is decoded to 16 kHz mono float32 so the whole
pipeline is deterministic regardless of the input container or sample rate."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def load(path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode any format ffmpeg understands to mono float32 at `sr`."""
    path = str(path)
    try:
        y, file_sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if file_sr != sr:
            import librosa

            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")
    except Exception:
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
               "-f", "f32le", "-"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        y = np.frombuffer(raw, dtype=np.float32).copy()
    return y.astype(np.float32)


def peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y if m == 0 else (y * (peak / m)).astype(np.float32)


def save(path: str | Path, y: np.ndarray, sr: int = SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1, 1), sr, subtype="PCM_16")


def crop(y: np.ndarray, start: float, end: float, sr: int = SR) -> np.ndarray:
    return y[int(round(start * sr)): int(round(end * sr))]
