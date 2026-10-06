"""Counterfactual repair: "what would this passage sound like without the flaw?"

For one finding, edit the participant's own recording so that the measured
cause is moved to the reference's value - and only that cause, only there:

  rushed / dragging  PSOLA time-scale the span to the reference articulation
                     rate (keeping the participant's overall habit)
  monotone           widen pitch excursions around the speaker's median to the
                     reference range
  trailing_off       restore the per-word level deficit against the reference
  mumbling           high-shelf boost until the >2 kHz energy share matches
  awkward_pause      shorten the pause to the reference pause
  filler             cut the held filler sound
  stutter            cut the repeated onset fragments
  pause_omission     reinsert the reference's pauses at the boundaries

Because the edit is an intervention on exactly the variable the explanation
names, re-analysing the repaired audio tests the explanation causally: if the
finding disappears and the score recovers, the stated cause was the cause
(tools/counterfactual_eval.py measures this over the dataset).
"""
from __future__ import annotations

import numpy as np

from .audio import SR
from .contrast import _bursts, pair_words
from .features import Analysis
from .inject import _psola

FADE = int(0.008 * SR)


def _splice(y, a, b, new):
    """y[:a] + new + y[b:], length exact. The first and last few ms of `new`
    are crossfaded with the original audio just inside the replaced span, so
    the joins are continuous for cuts, insertions and replacements alike."""
    a, b = int(a), int(b)
    new = np.asarray(new, np.float32).copy()
    y = y.astype(np.float32)
    n1 = min(FADE, len(new) // 2, len(y) - a)
    n2 = min(FADE, len(new) // 2, b)
    if n1 > 0:
        r = np.linspace(0, 1, n1, dtype=np.float32)
        new[:n1] = new[:n1] * r + y[a:a + n1] * (1 - r)
    if n2 > 0:
        r = np.linspace(1, 0, n2, dtype=np.float32)
        new[-n2:] = new[-n2:] * r + y[b - n2:b] * (1 - r)
    return np.concatenate([y[:a], new, y[b:]])


def _noise(y, n, seed):
    """Silence at the recording's own noise floor (quietest 10 % of 20 ms frames)."""
    k = len(y) // 320 * 320
    fr = y[:k].reshape(-1, 320)
    q = fr[np.argsort(np.sqrt((fr ** 2).mean(axis=1)))[: max(1, len(fr) // 10)]]
    return (np.random.default_rng(seed).standard_normal(int(n)) * float(q.std())).astype(np.float32)


def _metric(region, name):
    return next((m for m in region["metrics"] if m["name"] == name), None)


def _shelf(seg, gain_db, f0=1500.0, f1=2500.0):
    """High-shelf in the FFT domain, smooth ramp from f0 to f1."""
    spec = np.fft.rfft(seg.astype(np.float64))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    ramp = np.clip((f - f0) / (f1 - f0), 0, 1)
    out = np.fft.irfft(spec * 10 ** (gain_db * ramp / 20), len(seg))
    return out.astype(np.float32)


def repair(y: np.ndarray, ref: Analysis, par: Analysis, region: dict, glob: dict,
           ref_y: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Return (repaired audio, info). info has the edit applied in words and
    the snippet bounds in both timelines: orig=(s, e), fixed=(s, e)."""
    t = region["type"]
    s, e = float(region["start"]), float(region["end"])
    a, b = int(s * SR), int(e * SR)
    y = y.astype(np.float32)
    out, what = y, ""
    if region.get("scope") == "global":
        raise ValueError("global findings describe the whole reading; there is no single passage to repair")
    if t in ("rushed", "dragging"):
        m = _metric(region, "articulation_rate")
        target = m["reference"] * 2 ** glob.get("pace", {}).get("log2_ratio", 0.0)
        f = float(np.clip(m["participant"] / target, 0.5, 2.0))
        seg = _psola(y[a:b], dur_points=[(0.0, f), ((b - a) / SR, f)])
        out = _splice(y, a, b, seg)
        what = f"time-scaled {e - s:.2f} s by x{f:.2f} to {target:.2f} syll/s"
    elif t == "monotone":
        m = _metric(region, "pitch_range_p10_p90")
        target = m["reference"] * 2 ** glob.get("pitch", {}).get("log2_ratio", 0.0)
        k = float(np.clip(target / max(m["participant"], 0.3), 1.0, 4.0))
        a2, b2 = max(0, a - int(0.05 * SR)), min(len(y), b + int(0.05 * SR))
        med = par.ff.f0_median_hz

        def widen(ts, hz):
            return med * (hz / med) ** k
        seg = _psola(y[a2:b2], pitch_fn=widen)[: b2 - a2]
        seg = np.pad(seg, (0, b2 - a2 - len(seg)))
        out = _splice(y, a2, b2, seg)
        what = f"pitch excursions widened x{k:.2f} (range {m['participant']:.1f} -> {target:.1f} st)"
    elif t == "trailing_off":
        pairs = dict((j, i) for i, j in pair_words(ref, par))
        g = glob.get("volume", {}).get("offset_db", 0.0)
        ts, gains = [], []
        for j in range(region["word_start"], region["word_end"] + 1):
            if j not in pairs:
                continue
            d = par.wf[j].int_mean_db - ref.wf[pairs[j]].int_mean_db - g
            ts.append(0.5 * (par.words[j].start + par.words[j].end))
            gains.append(max(0.0, -d))
        if not ts:
            raise ValueError("no paired words in this span")
        tt = np.arange(a, b) / SR
        gain = np.interp(tt, ts, gains)
        ramp = min(len(gain) // 4, int(0.08 * SR))
        if ramp:
            gain[:ramp] *= np.linspace(0, 1, ramp)
            gain[-ramp:] *= np.linspace(1, 0, ramp)
        seg = y[a:b] * 10 ** (gain / 20)
        out = y.copy()
        out[a:b] = seg
        what = f"level restored by up to {max(gains):.1f} dB"
    elif t == "mumbling":
        m = _metric(region, "energy_above_2kHz")
        sh = m["participant"] / 100
        tgt = min(0.95, m["reference"] / 100 * 2 ** glob.get("clarity", {}).get("log2_ratio", 0.0))
        gdb = 0.0
        if 0 < sh < tgt:
            gdb = float(np.clip(10 * np.log10(tgt * (1 - sh) / (sh * (1 - tgt))), 0, 24))
        a2, b2 = max(0, a - int(0.03 * SR)), min(len(y), b + int(0.03 * SR))
        out = _splice(y, a2, b2, _shelf(y[a2:b2], gdb))
        what = f"high frequencies boosted {gdb:.1f} dB (share {sh * 100:.1f} % -> {tgt * 100:.1f} %)"
    elif t == "awkward_pause":
        m = _metric(region, "pause_length")
        keep = max(0.05, m["reference"])
        cut = (e - s) - keep
        if cut <= 0.02:
            raise ValueError("pause already at reference length")
        c0 = s + keep / 2
        out = _splice(y, int(c0 * SR), int((c0 + cut) * SR), np.zeros(0, np.float32))
        what = f"pause shortened {e - s:.2f} -> {keep:.2f} s"
    elif t == "filler":
        out = _splice(y, a, b, _noise(y, min(int(0.06 * SR), b - a), a))
        what = f"held filler sound removed ({e - s:.2f} s)"
    elif t == "stutter":
        fr = _bursts(par, max(0.0, s - 0.01), e)
        if len(fr) < 2:
            raise ValueError("no repeated onset found")
        c0, c1 = fr[0][0], fr[-1][0]
        out = _splice(y, int(c0 * SR), int(c1 * SR), np.zeros(0, np.float32))
        what = f"{len(fr) - 1} repeated onset fragment(s) removed ({c1 - c0:.2f} s)"
    elif t == "pause_omission":
        pairs = dict((j, i) for i, j in pair_words(ref, par))
        inserts = []
        for j in range(region["word_start"], region["word_end"]):
            i = pairs.get(j)
            if i is None or pairs.get(j + 1) != i + 1:
                continue
            pb, pp = ref.wf[i].pause_after, par.wf[j].pause_after
            if pb >= 0.18 and pp < 0.4 * pb:
                at = 0.5 * (par.words[j].end + par.words[j + 1].start)
                inserts.append((at, pb - pp))
        if not inserts:
            raise ValueError("no lost pause found")
        out = y
        for at, d in sorted(inserts, reverse=True):  # back to front keeps earlier times valid
            k = int(at * SR)
            out = _splice(out, k, k, _noise(y, d * SR, k))
        what = f"{len(inserts)} pause(s) reinserted ({sum(d for _, d in inserts):.2f} s)"
    elif t == "restart":
        # cut the second saying, keep a natural 60 ms joint
        keep = 0.06
        if e - s <= keep + 0.05:
            raise ValueError("repeated span too short")
        out = _splice(y, int((s + keep / 2) * SR), int((e - keep / 2) * SR), np.zeros(0, np.float32))
        what = f"the repeated “{region.get('text', '')}” removed ({e - s:.2f} s)"
    elif t == "skipped_word":
        # put the missing words back, taken from the reference reading
        if ref_y is None:
            raise ValueError("the reference audio is needed to restore skipped words")
        p2r = {j: i for i, j in pair_words(ref, par)}
        a_, b_ = region["word_start"], region["word_end"]
        miss = [j for j in range(a_ + 1, b_) if j in p2r]
        if not miss or a_ not in p2r or b_ not in p2r:
            raise ValueError("cannot place the missing words")
        r0, r1 = ref.words[p2r[miss[0]]].start, ref.words[p2r[miss[-1]]].end
        seg = ref_y[int(max(0, r0 - 0.02) * SR): int((r1 + 0.02) * SR)].astype(np.float32)
        # match the participant's level around the gap
        ctx = y[int(par.words[a_].start * SR): int(par.words[b_].end * SR)]
        g = (np.sqrt(np.mean(ctx ** 2)) + 1e-9) / (np.sqrt(np.mean(seg ** 2)) + 1e-9)
        gap = _noise(y, int(0.06 * SR), a)
        c0, c1 = par.words[a_].end, par.words[b_].start
        if c1 < c0:
            c0 = c1 = 0.5 * (c0 + c1)
        out = _splice(y, int(c0 * SR), int(c1 * SR), np.concatenate([gap, seg * float(np.clip(g, 0.25, 4)), gap]))
        words = " ".join(par.words[j].text for j in miss)
        what = f"the missing “{words}” restored from the reference reading"
    else:
        raise ValueError(f"no repair for {t}")
    shift = (len(out) - len(y)) / SR
    pad = 1.0
    info = {"type": t, "edit": what, "orig": (max(0.0, s - pad), min(len(y) / SR, e + pad)),
            "fixed": (max(0.0, s - pad), min(len(out) / SR, e + shift + pad)), "shift": round(shift, 3)}
    return out.astype(np.float32), info


def snippet(y: np.ndarray, span) -> np.ndarray:
    return y[int(span[0] * SR): int(span[1] * SR)]
