"""Flaw Injection Engine.

Turns one well-delivered ("gold") recording into a graded spectrum of flawed
recordings of the *same transcript*, and - because we perform every edit
ourselves - returns the exact ground-truth time span of each flaw in the new
timeline. All prosody edits use Praat's PSOLA resynthesis, which changes
timing and pitch while keeping the speaker's own voice.

Severity runs 1 (near-perfect, subtle) to 5 (egregious). Every random choice
comes from a seeded generator, so a dataset build is bit-for-bit reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import butter, sosfiltfilt

from .align import Word
from .audio import SR

FLAW_TYPES = ["rushed", "dragging", "monotone", "trailing_off", "pause_omission",
              "awkward_pause", "filler", "mumbling", "stutter"]

# severity 1..5 -> parameter
P = {
    "rushed":         [0.88, 0.80, 0.72, 0.64, 0.55],   # duration factor
    "dragging":       [1.12, 1.24, 1.36, 1.48, 1.62],
    "monotone":       [0.35, 0.55, 0.70, 0.85, 0.97],   # share of pitch excursion removed
    "trailing_off":   [4, 7, 10, 14, 18],                # dB drop reached at the end
    "pause_omission": [0.60, 0.45, 0.30, 0.15, 0.04],    # remaining share of each pause
    "awkward_pause":  [0.35, 0.60, 0.90, 1.30, 1.80],    # inserted silence (s)
    "filler":         [0.28, 0.36, 0.45, 0.55, 0.65],    # filler length (s)
    "mumbling":       [5000, 3500, 2500, 1800, 1200],    # low-pass cutoff (Hz)
    "stutter":        [1, 1, 2, 2, 3],                   # repetitions
}
COUNT = {"awkward_pause": [1, 1, 2, 2, 3], "filler": [1, 2, 2, 3, 4], "stutter": [1, 1, 1, 2, 2]}
SPAN = {"rushed": (6, 12), "dragging": (5, 10), "monotone": (8, 16), "trailing_off": (6, 12),
        "pause_omission": (10, 18), "mumbling": (5, 10)}

GAP_OPS = {"awkward_pause", "filler", "stutter"}


@dataclass
class Label:
    type: str
    severity: int
    start: float
    end: float
    word_start: int
    word_end: int
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------ primitives ---
def _psola(y, dur_points=None, pitch_fn=None, floor=60, ceil=500):
    """Praat overlap-add resynthesis. dur_points: [(t, factor)] (DurationTier),
    pitch_fn: maps (times, hz) -> new hz."""
    if len(y) < int(0.06 * SR):
        return y.copy()
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    manip = call(snd, "To Manipulation", 0.01, floor, ceil)
    if dur_points:
        dt = call("Create DurationTier", "d", 0, snd.duration)
        for t, f in dur_points:
            call(dt, "Add point", float(t), float(f))
        call([manip, dt], "Replace duration tier")
    if pitch_fn is not None:
        pt = call(manip, "Extract pitch tier")
        n = call(pt, "Get number of points")
        if n > 2:
            ts = np.array([call(pt, "Get time from index", i + 1) for i in range(n)])
            hz = np.array([call(pt, "Get value at index", i + 1) for i in range(n)])
            new = pitch_fn(ts, hz)
            call(pt, "Remove points between", 0, snd.duration)
            for t, v in zip(ts, new):
                call(pt, "Add point", float(t), float(v))
            call([manip, pt], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    return out.values[0].astype(np.float32)


def _fade(y, ms=8):
    n = min(len(y) // 2, int(ms / 1000 * SR))
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        y = y.copy()
        y[:n] *= r
        y[-n:] *= r[::-1]
    return y


def _silence(sec, noise_ref=None, rng=None):
    """Silence carrying the recording's own noise floor, so inserted pauses are
    not given away by digital zero."""
    n = int(round(sec * SR))
    if noise_ref is None or len(noise_ref) < 400 or rng is None:
        return np.zeros(n, dtype=np.float32)
    std = float(np.std(noise_ref))
    return (rng.standard_normal(n) * std).astype(np.float32)


# ----------------------------------------------------------------- engine ---
class Injector:
    def __init__(self, y: np.ndarray, words: list[Word], f0_median_hz: float = 150.0, seed: int = 0):
        self.y = y.astype(np.float32)
        words = [Word(**w.to_dict()) for w in words]
        for k in range(len(words) - 1):  # guarantee non-overlapping spans
            if words[k].end > words[k + 1].start:
                mid = round((words[k].end + words[k + 1].start) / 2, 3)
                words[k].end, words[k + 1].start = mid, mid
        self.words = words
        self.rng = np.random.default_rng(seed)
        self.floor = max(50, f0_median_hz * 0.55)
        self.ceil = min(800, f0_median_hz * 2.4)
        self.f0_med = f0_median_hz
        # cut points: middle of each inter-word gap
        self.cut = [0.0] + [(words[k - 1].end + words[k].start) / 2 for k in range(1, len(words))] + [len(y) / SR]
        self.noise = self._noise_floor()

    def _noise_floor(self):
        gaps = [(self.words[k].end, self.words[k + 1].start) for k in range(len(self.words) - 1)]
        gaps = [g for g in gaps if g[1] - g[0] > 0.15]
        if not gaps:
            return None
        a, b = max(gaps, key=lambda g: g[1] - g[0])
        return self.y[int((a + 0.03) * SR): int((b - 0.03) * SR)]

    # -- choosing where flaws go
    def pick_span(self, ftype, taken):
        lo, hi = SPAN[ftype]
        n = int(self.rng.integers(lo, hi + 1))
        n = min(n, len(self.words) - 2)
        for _ in range(200):
            a = int(self.rng.integers(1, len(self.words) - n))
            b = a + n - 1
            if ftype == "pause_omission":
                if sum(1 for k in range(a, b) if self.words[k].punct_after) < 2:
                    continue
            if all(b < s - 1 or a > e + 1 for s, e in taken):
                return a, b
        return None

    def pick_gaps(self, ftype, count, taken):
        cands = [k for k in range(1, len(self.words) - 2)
                 if (ftype == "stutter" or not self.words[k].punct_after)
                 and all(k < s - 1 or k > e + 1 for s, e in taken)]
        if ftype == "stutter":
            cands = [k for k in cands if self.words[k + 1].end - self.words[k + 1].start > 0.18]
        if len(cands) < count:
            return []
        picks = sorted(self.rng.choice(cands, size=count, replace=False).tolist())
        return picks

    # -- building
    def apply(self, specs: list[tuple[str, int]]):
        """specs: [(flaw_type, severity)]. Returns (audio, new_words, labels)."""
        taken: list[tuple[int, int]] = []
        ops = []
        for ftype, sev in specs:
            if ftype in GAP_OPS:
                ks = self.pick_gaps(ftype, COUNT[ftype][sev - 1], taken)
                for k in ks:
                    ops.append((ftype, sev, k, k))
                    taken.append((k, k + 1))
            else:
                span = self.pick_span(ftype, taken)
                if span:
                    ops.append((ftype, sev) + span)
                    taken.append(span)
        return self._render(sorted(ops, key=lambda o: o[2]))

    def _render(self, ops):
        W = self.words
        pieces = []          # audio chunks
        tmap_old, tmap_new = [0.0], [0.0]  # piecewise-linear time map
        pos_old, pos_new = 0.0, 0.0
        pending = []         # labels in word indices + markers resolved after mapping

        def keep_until(t):
            nonlocal pos_old, pos_new
            if t > pos_old:
                pieces.append(self.y[int(pos_old * SR): int(t * SR)])
                pos_new += (int(t * SR) - int(pos_old * SR)) / SR
                pos_old = t
                tmap_old.append(pos_old)
                tmap_new.append(pos_new)

        def emit(chunk, old_pts, new_pts):
            nonlocal pos_old, pos_new
            pieces.append(chunk)
            for o, n in zip(old_pts, new_pts):
                tmap_old.append(pos_old + o)
                tmap_new.append(pos_new + n)
            pos_old += old_pts[-1]
            pos_new += len(chunk) / SR

        for ftype, sev, a, b in ops:
            p = P[ftype][sev - 1]
            if ftype in GAP_OPS:
                k = a
                if ftype == "stutter":
                    w = W[k + 1]
                    keep_until(w.start)
                    seg = self.y[int(w.start * SR): int((w.start + min(0.15, (w.end - w.start) * 0.4)) * SR)]
                    ins = []
                    for _ in range(p):
                        ins += [_fade(seg, 5), _silence(0.07 + 0.03 * self.rng.random(), self.noise, self.rng)]
                    chunk = np.concatenate(ins)
                    pieces.append(chunk)
                    pos_new += len(chunk) / SR
                    tmap_old.append(pos_old)
                    tmap_new.append(pos_new)
                    pending.append(Label(ftype, sev, -1, -1, k + 1, k + 1, {"repetitions": p}))
                    pending[-1].params["_t"] = (pos_new - len(chunk) / SR, None)
                    continue
                gap_mid = (W[k].end + W[k + 1].start) / 2
                keep_until(gap_mid)
                if ftype == "awkward_pause":
                    chunk = _silence(p * (0.85 + 0.3 * self.rng.random()), self.noise, self.rng)
                    params = {"inserted_s": round(len(chunk) / SR, 3)}
                else:
                    chunk = self._filler(p)
                    pad = _silence(0.08, self.noise, self.rng)
                    chunk = np.concatenate([pad, chunk, pad])
                    params = {"filler_s": round(p, 3)}
                pieces.append(chunk)
                pos_new += len(chunk) / SR
                tmap_old.append(pos_old)
                tmap_new.append(pos_new)
                pending.append(Label(ftype, sev, -1, -1, k, k + 1, params))
                continue

            c0, c1 = self.cut[a], self.cut[b + 1]
            keep_until(c0)
            seg = self.y[int(c0 * SR): int(c1 * SR)]
            L = len(seg) / SR
            if ftype in ("rushed", "dragging"):
                # words scaled by p; pauses scaled harder (p^1.6) - rushing eats pauses first
                pts, prev = [], 0.0
                gp = p ** 1.6
                for k in range(a, b + 1):
                    ws, we = W[k].start - c0, W[k].end - c0
                    pts += [(max(0, ws - 0.001), gp), (ws, p), (we, p), (we + 0.001, gp)]
                pts = sorted(set((round(t, 4), f) for t, f in pts if 0 <= t <= L))
                out = _psola(seg, pts, floor=self.floor, ceil=self.ceil)
                old, new = self._dur_map(pts, L, len(out) / SR)
                emit(out, old, new)
                params = {"duration_factor": p}
            elif ftype == "monotone":
                def flat(ts, hz, keep=1 - p):
                    st = 12 * np.log2(hz / self.f0_med)
                    c = np.median(st)
                    return self.f0_med * 2 ** ((c + (st - c) * keep) / 12)
                out = _psola(seg, pitch_fn=flat, floor=self.floor, ceil=self.ceil)
                out = out[:len(seg)] if len(out) >= len(seg) else np.pad(out, (0, len(seg) - len(out)))
                emit(out, [L], [L])
                params = {"excursion_removed": p}
            elif ftype == "trailing_off":
                n = len(seg)
                ramp_len = int(n * 0.6)
                g_db = np.concatenate([np.linspace(0, -p, ramp_len), np.full(n - ramp_len, -p)])
                emit((seg * 10 ** (g_db / 20)).astype(np.float32), [L], [L])
                params = {"drop_db": p}
            elif ftype == "mumbling":
                sos = butter(6, p, btype="low", fs=SR, output="sos")
                out = sosfiltfilt(sos, seg).astype(np.float32) * 10 ** (-(sev * 1.0) / 20)
                emit(_crossfade_edges(seg, out), [L], [L])
                params = {"lowpass_hz": p}
            elif ftype == "pause_omission":
                parts, old_pts, new_pts, t_new = [], [], [], 0.0
                cur = c0
                n_cut = 0
                for k in range(a, b):
                    if not W[k].punct_after:
                        continue
                    g0, g1 = W[k].end + 0.02, W[k + 1].start - 0.02
                    if g1 - g0 < 0.06:
                        continue
                    keep = max(0.02, (g1 - g0) * p)
                    parts.append(self.y[int(cur * SR): int(g0 * SR)])
                    t_new += (int(g0 * SR) - int(cur * SR)) / SR
                    old_pts.append(g0 - c0); new_pts.append(t_new)
                    mid = (g0 + g1) / 2
                    parts.append(_fade(self.y[int((mid - keep / 2) * SR): int((mid + keep / 2) * SR)], 3))
                    t_new += (int((mid + keep / 2) * SR) - int((mid - keep / 2) * SR)) / SR
                    old_pts.append(g1 - c0); new_pts.append(t_new)
                    parts.append(np.zeros(0, np.float32))
                    cur = g1
                    n_cut += 1
                parts.append(self.y[int(cur * SR): int(c1 * SR)])
                t_new += (int(c1 * SR) - int(cur * SR)) / SR
                old_pts.append(L); new_pts.append(t_new)
                emit(np.concatenate(parts), old_pts, new_pts)
                params = {"pause_kept": p, "pauses_shortened": n_cut}
            pending.append(Label(ftype, sev, -1, -1, a, b, params))
        keep_until(len(self.y) / SR)

        out = np.concatenate(pieces).astype(np.float32)
        to = np.array(tmap_old)
        tn = np.array(tmap_new)
        order = np.argsort(to, kind="stable")
        to, tn = to[order], tn[order]

        def m(t, side="left"):
            # side matters at insertion points (same old time maps to 2 new times)
            idx = np.searchsorted(to, t, side=side)
            idx = np.clip(idx, 1, len(to) - 1)
            o0, o1, n0, n1 = to[idx - 1], to[idx], tn[idx - 1], tn[idx]
            return float(n0 + (t - o0) * ((n1 - n0) / (o1 - o0) if o1 > o0 else 0))

        new_words = [Word(w.i, w.text, w.norm, round(m(w.start, "right"), 3), round(m(w.end, "left"), 3),
                          w.conf, w.punct_after) for w in W]
        labels = []
        for lb in pending:
            if lb.type in ("awkward_pause", "filler"):
                lb.start = new_words[lb.word_start].end
                lb.end = new_words[lb.word_end].start
            elif lb.type == "stutter":
                lb.start = round(lb.params.pop("_t")[0], 3)
                lb.end = new_words[lb.word_start].end
            else:
                lb.start = new_words[lb.word_start].start
                lb.end = new_words[lb.word_end].end
            lb.start, lb.end = round(lb.start, 3), round(lb.end, 3)
            labels.append(lb)
        return out, new_words, labels

    @staticmethod
    def _dur_map(pts, L, out_len, res=0.001):
        """Integrate Praat's piecewise-linear DurationTier to map old->new time,
        then rescale by the (tiny) difference to the true resynthesized length."""
        ts = np.arange(0, L + res, res)
        pt_t = np.array([t for t, _ in pts])
        pt_f = np.array([f for _, f in pts])
        f = np.interp(ts, pt_t, pt_f)
        new = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * res)])
        if new[-1] > 0:
            new *= out_len / new[-1]
        step = max(1, len(ts) // 400)
        return ts[::step].tolist() + [L], new[::step].tolist() + [out_len]

    def _filler(self, dur):
        """Speaker-matched "uh"/"um": take the steadiest voiced 90 ms of this
        speaker's own audio (highest periodicity, lowest F0 movement), stretch it
        with PSOLA to `dur`, give it a gently falling pitch, and for "um" close it
        with a nasal (low-passed) tail."""
        if not hasattr(self, "_vowel"):
            snd = parselmouth.Sound(self.y.astype(np.float64), sampling_frequency=SR)
            pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=self.floor, pitch_ceiling=self.ceil)
            hnr = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=self.floor)
            f = pitch.selected_array["frequency"]
            hv = np.interp(pitch.xs(), hnr.xs(), hnr.values[0])
            best, best_t = -1e9, None
            for i in range(len(f) - 9):
                win = f[i:i + 9]
                if (win > 0).all():
                    score = hv[i:i + 9].mean() - 3 * np.std(12 * np.log2(win / win.mean()))
                    if score > best:
                        best, best_t = score, pitch.xs()[i]
            t0 = best_t if best_t is not None else self.words[len(self.words) // 2].start
            self._vowel = self.y[int(t0 * SR): int((t0 + 0.09) * SR)]
        v = self._vowel
        factor = dur / (len(v) / SR)
        L = len(v) / SR

        def fall(ts, hz):
            return np.full_like(hz, self.f0_med * 0.97) * (1 - 0.06 * ts / max(ts.max(), 1e-3))
        out = _psola(v, [(0, factor), (L, factor)], pitch_fn=fall, floor=self.floor, ceil=self.ceil)
        if self.rng.random() < 0.5:  # "um"
            n = len(out)
            tail = int(n * 0.4)
            sos = butter(4, 450, btype="low", fs=SR, output="sos")
            nasal = sosfiltfilt(sos, out[-tail:]) * 1.4
            xf = np.linspace(0, 1, tail)
            out = np.concatenate([out[:-tail], out[-tail:] * (1 - xf) + nasal * xf]).astype(np.float32)
        env = np.ones(len(out), np.float32)
        a = int(0.04 * SR)
        env[:a] = np.linspace(0, 1, a)
        env[-a * 2:] = np.linspace(1, 0, a * 2)
        rms_ref = np.sqrt(np.mean(v ** 2)) + 1e-9
        return (out * env * 0.8 * rms_ref / (np.sqrt(np.mean(out ** 2)) + 1e-9)).astype(np.float32)


def _crossfade_edges(orig, new, ms=15):
    n = min(len(orig) // 3, int(ms / 1000 * SR))
    out = new.copy()
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        out[:n] = orig[:n] * (1 - r) + new[:n] * r
        out[-n:] = new[-n:] * (1 - r) + orig[-n:] * r
    return out
