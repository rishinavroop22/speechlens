"""Forced alignment of a known transcript to audio.

Primary method: wav2vec2-base-960h (ONNX, CPU) emits per-frame character
log-probabilities; a CTC Viterbi pass finds the single most likely path that
spells the transcript exactly, giving each word a start/end frame (20 ms).
Boundaries are then refined against the energy envelope so word ends include
their decaying tail and pauses are measured from real silence.

No torch dependency: inference is onnxruntime, the Viterbi is numpy.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import SR
from .text import Token, tokenize

FRAME = 0.02  # wav2vec2 output stride (320 samples @ 16 kHz)
MODEL_DIR = Path(os.environ.get("SPEECHLENS_MODEL_DIR",
                                Path(__file__).resolve().parent.parent / "models" / "wav2vec2"))


@dataclass
class Word:
    i: int
    text: str
    norm: str
    start: float
    end: float
    conf: float               # mean posterior of the intended characters (0-1)
    punct_after: str | None

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- model ---
@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    onnx = MODEL_DIR / "onnx" / "model.onnx"
    if not onnx.exists():
        onnx = MODEL_DIR / "model.onnx"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("SPEECHLENS_THREADS", max(1, os.cpu_count() or 1)))
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"])
    vocab = json.loads((MODEL_DIR / "vocab.json").read_text())
    return sess, vocab


def model_available() -> bool:
    return (MODEL_DIR / "vocab.json").exists() and (
        (MODEL_DIR / "onnx" / "model.onnx").exists() or (MODEL_DIR / "model.onnx").exists())


def _log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def emissions(y: np.ndarray, win: float = 20.0, ctx: float = 1.0) -> np.ndarray:
    """Frame log-probs [T, V]. Long audio is processed in 20 s windows with 1 s
    context each side so memory stays flat and output is identical run to run."""
    sess, _ = _session()
    name = sess.get_inputs()[0].name
    hop = int(win * SR)
    pad = int(ctx * SR)
    n_frames = int(np.ceil(len(y) / (FRAME * SR)))
    out = []
    for s in range(0, len(y), hop):
        a, b = max(0, s - pad), min(len(y), s + hop + pad)
        seg = y[a:b]
        seg = (seg - seg.mean()) / (seg.std() + 1e-7)
        logits = sess.run(None, {name: seg[None, :].astype(np.float32)})[0][0]
        f0 = int(round((s - a) / (FRAME * SR)))
        f1 = f0 + int(round((min(s + hop, len(y)) - s) / (FRAME * SR)))
        out.append(logits[f0:min(f1, len(logits))])
    lp = _log_softmax(np.concatenate(out, axis=0).astype(np.float64))
    if len(lp) < n_frames:  # pad tail with blanks
        tail = np.full((n_frames - len(lp), lp.shape[1]), -30.0)
        tail[:, 0] = 0.0
        lp = np.vstack([lp, tail])
    return lp[:n_frames]


# -------------------------------------------------------------- viterbi ---
def ctc_viterbi(lp: np.ndarray, targets: list[int], blank: int = 0) -> np.ndarray:
    """Return, for each frame, the index into `targets` it is assigned to
    (-1 for blank). Exact CTC forced alignment (no beam, deterministic)."""
    T = lp.shape[0]
    L = len(targets)
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = targets
    can_skip = np.zeros(S, dtype=bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]
    if T < L + int(np.sum(~can_skip[3::2])):
        raise ValueError(f"audio too short for transcript ({T} frames, {L} tokens)")
    NEG = -1e30
    alpha = np.full(S, NEG)
    alpha[0] = lp[0, ext[0]]
    alpha[1] = lp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        stay = alpha
        step = np.concatenate([[NEG], alpha[:-1]])
        skip = np.where(can_skip, np.concatenate([[NEG, NEG], alpha[:-2]]), NEG)
        cand = np.stack([stay, step, skip])
        arg = np.argmax(cand, axis=0)
        alpha = cand[arg, np.arange(S)] + lp[t, ext]
        back[t] = arg
    s = int(S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2)
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    return np.where(path % 2 == 1, (path - 1) // 2, -1)


# ------------------------------------------------------------- refining ---
def _rms_db(y: np.ndarray, hop: int = 160, win: int = 400) -> np.ndarray:
    import librosa

    r = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    ref = np.percentile(r, 95) + 1e-9
    return 20 * np.log10(r / ref + 1e-9)


def refine(words: list[Word], y: np.ndarray, thresh_db: float = -32.0) -> list[Word]:
    """Snap boundaries to the energy envelope (10 ms resolution). CTC spikes
    mark onsets well but cut word tails short; we extend each word while the
    signal stays above `thresh_db` (re. 95th pct), never crossing neighbours."""
    db = _rms_db(y)
    hop = 0.01
    n = len(db)
    for k, w in enumerate(words):
        lo = words[k - 1].end if k else 0.0
        hi = words[k + 1].start if k + 1 < len(words) else n * hop
        a, b = int(w.start / hop), int(w.end / hop)
        while a - 1 > int(lo / hop) and db[a - 1] > thresh_db and (w.start - (a - 1) * hop) < 0.12:
            a -= 1
        while b < min(n, int(hi / hop)) and db[b] > thresh_db and ((b * hop) - w.end) < 0.35:
            b += 1
        w.start, w.end = round(a * hop, 3), round(max(b, a + 2) * hop, 3)
    return words


# ---------------------------------------------------------------- public ---
def align(y: np.ndarray, transcript: str | list[Token]) -> tuple[list[Word], str]:
    tokens = tokenize(transcript) if isinstance(transcript, str) else transcript
    if not model_available():
        return fallback_align(y, tokens), "fallback-energy"
    _, vocab = _session()
    blank = vocab.get("<pad>", 0)
    sep = vocab["|"]
    unk = vocab.get("<unk>", 3)
    targets, owner = [], []
    for k, tok in enumerate(tokens):
        if k:
            targets.append(sep)
            owner.append(-1)
        for ch in tok.norm:
            targets.append(vocab.get(ch, unk))
            owner.append(k)
    lp = emissions(y)
    frame_tgt = ctc_viterbi(lp, targets, blank)
    owner = np.array(owner)
    words = []
    for k, tok in enumerate(tokens):
        idx = np.nonzero(owner == k)[0]
        frames = np.nonzero(np.isin(frame_tgt, idx))[0]
        f0, f1 = int(frames.min()), int(frames.max()) + 1
        tgt_ids = np.array(targets)[frame_tgt[frames]]
        conf = float(np.exp(lp[frames, tgt_ids]).mean())
        words.append(Word(k, tok.text, tok.norm, round(f0 * FRAME, 3), round(f1 * FRAME, 3),
                          round(conf, 4), tok.punct_after))
    return refine(words, y), "wav2vec2-ctc"


def fallback_align(y: np.ndarray, tokens: list[Token]) -> list[Word]:
    """Model-free approximation used only when the ONNX model is absent:
    distributes words over detected speech in proportion to syllable count."""
    from .text import count_syllables

    db = _rms_db(y)
    speech = db > -35
    hop = 0.01
    syl = np.array([count_syllables(t.norm) for t in tokens], dtype=float)
    frames = np.nonzero(speech)[0]
    if len(frames) == 0:
        frames = np.arange(len(db))
    cum = np.concatenate([[0], np.cumsum(syl)]) / syl.sum() * (len(frames) - 1)
    words = []
    for k, t in enumerate(tokens):
        a = frames[int(cum[k])]
        b = frames[min(len(frames) - 1, int(cum[k + 1]))] + 1
        words.append(Word(k, t.text, t.norm, round(a * hop, 3), round(b * hop, 3), 0.0, t.punct_after))
    for k in range(len(words) - 1):
        words[k].end = min(words[k].end, words[k + 1].start)
    return words
