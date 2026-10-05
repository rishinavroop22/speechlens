"""Fast unit tests (no alignment model needed). Run: python -m pytest -q"""
import numpy as np
import pytest

from speechlens import audio
from speechlens.align import Word, ctc_viterbi
from speechlens.contrast import _runs, _sev
from speechlens.inject import Injector, FLAW_TYPES
from speechlens.text import count_syllables, tokenize

SR = audio.SR


def test_tokenize_numbers_dates_dashes():
    toks = tokenize("Yesterday, December 7, 1941—a date which will live in infamy.")
    norms = [t.norm for t in toks]
    assert norms[:4] == ["YESTERDAY", "DECEMBER", "SEVENTH", "NINETEEN|FORTY|ONE"]
    assert toks[0].punct_after == "comma"
    assert toks[3].punct_after == "clause"
    assert toks[-1].punct_after == "sentence"


def test_syllables():
    assert count_syllables("COUNTRY") == 2
    assert count_syllables("AMERICANS") == 4
    assert count_syllables("THE") == 1


def test_ctc_viterbi_exact_path():
    # 3 symbols + blank; emissions spell "1 2 2" with a blank between repeats
    T, V = 12, 4
    lp = np.full((T, V), -10.0)
    seq = [0, 1, 1, 0, 2, 2, 0, 0, 2, 2, 2, 0]
    for t, c in enumerate(seq):
        lp[t, c] = 0.0
    path = ctc_viterbi(lp, [1, 2, 2], blank=0)
    assert list(path) == [-1, 0, 0, -1, 1, 1, -1, -1, 2, 2, 2, -1]


def test_ctc_viterbi_rejects_impossible():
    with pytest.raises(ValueError):
        ctc_viterbi(np.zeros((2, 4)), [1, 2, 3])


def test_runs_closes_single_holes():
    assert _runs([1, 1, 0, 1, 1, 0, 0, 0, 1], min_len=3, hole=1) == [(0, 4)]


def test_severity_mapping_monotone():
    tbl = [0.88, 0.80, 0.72, 0.64, 0.55]
    sev = [_sev(v, tbl, increasing=False) for v in (0.9, 0.81, 0.7, 0.6, 0.5)]
    assert sev == [1, 2, 3, 4, 5]


def _toy_speech(n_words=24, seed=0):
    """Synthetic 'speech': voiced harmonic bursts with pitch movement, silence between."""
    rng = np.random.default_rng(seed)
    parts, words, t = [], [], 0.2
    parts.append(np.zeros(int(0.2 * SR), np.float32))
    for i in range(n_words):
        d = rng.uniform(0.25, 0.45)
        tt = np.arange(int(d * SR)) / SR
        f0 = 140 * 2 ** (rng.uniform(-3, 3) / 12) * (1 + 0.05 * np.sin(2 * np.pi * 3 * tt))
        ph = 2 * np.pi * np.cumsum(f0) / SR
        y = sum(np.sin(k * ph) / k for k in range(1, 8)) * np.hanning(len(tt)) * 0.3
        parts.append(y.astype(np.float32))
        punct = "comma" if i % 6 == 5 else None
        words.append(Word(i, f"w{i}", "BA", round(t, 3), round(t + d, 3), 1.0, punct))
        gap = rng.uniform(0.25, 0.4) if punct else rng.uniform(0.06, 0.12)
        parts.append((rng.standard_normal(int(gap * SR)) * 1e-3).astype(np.float32))
        t += d + gap
    return np.concatenate(parts), words


@pytest.mark.parametrize("ftype", FLAW_TYPES)
def test_injection_labels_are_consistent(ftype):
    y, words = _toy_speech()
    inj = Injector(y, words, 140.0, seed=3)
    out, nw, labels = inj.apply([(ftype, 4)])
    dur = len(out) / SR
    assert labels, f"{ftype}: no label produced"
    for lb in labels:
        assert 0 <= lb.start < lb.end <= dur + 1e-6
        assert lb.type == ftype and lb.severity == 4
    for a, b in zip(nw, nw[1:]):
        assert a.start <= a.end <= b.start + 1e-6


def test_injection_is_deterministic():
    y, words = _toy_speech()
    a = Injector(y, words, 140.0, seed=7).apply([("rushed", 3), ("filler", 3)])
    b = Injector(y, words, 140.0, seed=7).apply([("rushed", 3), ("filler", 3)])
    assert np.array_equal(a[0], b[0])
    assert [l.to_dict() for l in a[2]] == [l.to_dict() for l in b[2]]


def test_rushed_shortens_and_dragging_lengthens():
    y, words = _toy_speech()
    r, _, _ = Injector(y, words, 140.0, seed=1).apply([("rushed", 5)])
    d, _, _ = Injector(y, words, 140.0, seed=1).apply([("dragging", 5)])
    assert len(r) < len(y) < len(d)
