"""Contrastive analysis: compare a participant reading against a reference
reading of the same transcript, word by word, and ground every delivery flaw
to an exact time span with the numbers that justify it.

Method
------
1. Words of the two readings are paired by transcript position (difflib on
   the normalized spellings, so a skipped or added word does not derail it).
2. For each pair we compute six contrast signals (pace, pitch variety,
   loudness, articulation, pause, disfluency). Each continuous signal s_j is
   split into a *global* offset g = median_j(s_j) - the speaker's overall
   habit, reported separately - and a *local* deviation l_j = s_j - g that
   pinpoints where delivery departs from the reference.
3. Local deviations are scored in units of the reference speaker's own
   word-to-word variability (robust z = l / (1.4826 * MAD)), smoothed over
   neighbouring words, thresholded, and merged into contiguous regions.
4. Each region carries its metric values, deltas, z-score and the threshold
   that fired, which the explainer turns into a causal sentence.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field, asdict

import numpy as np

from .features import Analysis, HOP

DIMENSION = {"rushed": "pace", "dragging": "pace", "pause_omission": "pausing", "awkward_pause": "pausing",
             "filler": "fluency", "stutter": "fluency", "monotone": "pitch_variety",
             "trailing_off": "volume", "mumbling": "clarity"}

# local thresholds (in natural units) - see docs/methodology.md
TH = {
    "pace_log2": 0.20,          # >= 15 % faster / slower than the reference, locally
    "pitch_log2": -0.55,        # pitch spread < 68 % of reference
    "vol_db": -3.5,             # quieter than reference by > 3.5 dB (speaker-relative)
    "clar_log2": -0.75,         # high-band articulation energy < 59 % of reference
    "pause_keep": 0.40,         # a reference pause >= 180 ms shrunk to < 40 %
    "pause_extra_s": 0.30,      # >= 300 ms longer than the reference pause, mid-phrase
    "pause_extra_punct_s": 1.20,  # at punctuation, only a pause >= 1.2 s longer counts
    "filled_s": 0.15,           # >= 150 ms of voicing inside a pause
}
# global (whole-reading) thresholds
TH_GLOBAL = {"pace_log2": 0.25, "pitch_log2": -0.6, "clar_log2": -0.8}


@dataclass
class Metric:
    name: str
    unit: str
    reference: float
    participant: float
    delta: float
    z: float | None = None
    threshold: str = ""

    def to_dict(self):
        d = asdict(self)
        for k in ("reference", "participant", "delta", "z"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        return d


@dataclass
class Region:
    type: str
    start: float
    end: float
    word_start: int            # participant word indices
    word_end: int
    severity: int              # 1..5 estimate
    confidence: float          # 0..1
    metrics: list[Metric] = field(default_factory=list)
    scope: str = "local"       # "local" | "global"
    ref_start: float | None = None
    ref_end: float | None = None
    text: str = ""
    explanation: str = ""
    magnitude: float | None = None

    def __post_init__(self):
        if isinstance(self.severity, Sev):
            self.magnitude = round(self.severity.magnitude, 4)
        self.severity = int(self.severity)

    @property
    def dimension(self):
        return DIMENSION[self.type]

    def to_dict(self):
        d = asdict(self)
        d["dimension"] = self.dimension
        d["metrics"] = [m.to_dict() for m in self.metrics]
        return d


# ------------------------------------------------------------------ utils ---
def pair_words(ref: Analysis, par: Analysis) -> list[tuple[int, int]]:
    a = [w.norm for w in ref.words]
    b = [w.norm for w in par.words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    pairs = []
    for blk in sm.get_matching_blocks():
        pairs += [(blk.a + k, blk.b + k) for k in range(blk.size)]
    return pairs


def _robust_spread(x, floor):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return floor
    return max(floor, 1.4826 * float(np.median(np.abs(x - np.median(x)))))


def _smooth(x, k=3):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    h = k // 2
    for i in range(len(x)):
        w = x[max(0, i - h): i + h + 1]
        w = w[np.isfinite(w)]
        out[i] = np.mean(w) if len(w) else np.nan
    return out


def _runs(flags, min_len=1, hole=1):
    """Contiguous True runs, closing gaps of <= `hole` False entries."""
    f = np.asarray(flags, bool).copy()
    idx = np.nonzero(f)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= hole + 1:
            f[a:b] = True
    runs, start = [], None
    for i, v in enumerate(np.r_[f, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    return runs


def _window(vals_num, vals_den, j, h):
    a, b = max(0, j - h), j + h + 1
    num, den = np.nansum(vals_num[a:b]), np.nansum(vals_den[a:b])
    return num / den if den > 0 else np.nan


def _f0_spread(an: Analysis, wi: list[int]) -> float:
    fr = [an.ff.f0_st[an.ff.at(an.words[i].start, an.words[i].end)] for i in wi]
    fr = np.concatenate(fr) if fr else np.array([])
    fr = fr[np.isfinite(fr)]
    if len(fr) < 8:
        return np.nan
    # drop octave errors (a tracker jumping 12 st) before measuring the range
    fr = fr[np.abs(fr - np.median(fr)) < 7]
    if len(fr) < 8:
        return np.nan
    return float(np.percentile(fr, 90) - np.percentile(fr, 10))


def _levels(an: Analysis):
    """(noise floor, speech peak) in the recording's own dB scale: the 10th and
    90th percentiles of intensity. Thresholds placed between them work for a
    studio recording and for an outdoor one with crowd noise alike."""
    if not hasattr(an, "_lv"):
        x = an.ff.intensity_db
        an._lv = (float(np.percentile(x, 10)), float(np.percentile(x, 90)))
    return an._lv


def _bursts(an: Analysis, a: float, b: float, frac=0.45):
    """Energy bursts inside [a, b] (onset repetitions, fillers): runs above a
    point `frac` of the way from the noise floor to the speech peak."""
    if b - a < 0.06:
        return []
    lo, hi = _levels(an)
    thr = lo + frac * (hi - lo)
    sl = an.ff.at(a, b)
    on = an.ff.intensity_db[sl] > thr
    segs, s = [], None
    for i, v in enumerate(np.r_[on, False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= 3:
                segs.append((a + s * HOP, a + i * HOP))
            s = None
    return segs


def _steady_mask(an: Analysis) -> np.ndarray:
    """Frames of held, unchanging voicing: voiced, loud enough, pitch within
    +/-0.5 st over 100 ms, and spectral change in the speaker's calmest 30 %.
    A filled pause ("uhhh") is exactly this; running speech rarely is for long."""
    if hasattr(an, "_steady"):
        return an._steady
    ff = an.ff
    lo, hi = _levels(an)
    voiced = np.isfinite(ff.f0_st) & (ff.intensity_db > lo + 0.4 * (hi - lo))
    f0 = np.where(np.isfinite(ff.f0_st), ff.f0_st, 0.0)
    k = 10
    pad = lambda x: np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad(f0), k)
    vwin = np.lib.stride_tricks.sliding_window_view(pad(voiced.astype(float)), k)
    f0_std = win.std(axis=1)
    dm = np.r_[0, np.linalg.norm(np.diff(ff.mfcc, axis=1), axis=0)]
    dm_s = np.convolve(dm, np.ones(5) / 5, mode="same")
    calm = np.percentile(dm_s[voiced], 30) if voiced.any() else 0
    m = voiced & (vwin.mean(axis=1) > 0.95) & (f0_std < 0.5) & (dm_s <= calm)
    an._steady = m
    return m


def _run_near(mask, a, b, t, min_len):
    """Among True runs in [a, b] at least `min_len` s long, the one whose
    centre is nearest `t` (falls back to the longest run)."""
    sl = slice(int(a / HOP), int(b / HOP))
    x = np.r_[mask[sl], False]
    off = int(a / HOP)
    runs, run = [], 0
    for i, v in enumerate(x):
        if v:
            run += 1
        elif run:
            runs.append((run * HOP, (off + i - run) * HOP, (off + i) * HOP))
            run = 0
    long = [r for r in runs if r[0] >= min_len]
    if long:
        return min(long, key=lambda r: abs(0.5 * (r[1] + r[2]) - t))
    return max(runs, default=(0.0, 0.0, 0.0))


def _longest_run(mask, a, b):
    """Longest True run inside [a, b] s; returns (length_s, start_s, end_s)."""
    sl = slice(int(a / HOP), int(b / HOP))
    x = mask[sl]
    best = (0, 0, 0)
    run = 0
    for i, v in enumerate(np.r_[x, False]):
        if v:
            run += 1
        else:
            if run > best[0]:
                best = (run, i - run, i)
            run = 0
    off = int(a / HOP)
    return best[0] * HOP, (off + best[1]) * HOP, (off + best[2]) * HOP


def _onset_similarity(an: Analysis, burst, word_start, dur=0.12):
    """Cosine similarity of mean MFCC (c1..c12) of a burst vs the next word's
    first 120 ms. A repeated onset ("c-c-country") scores high; a neutral
    filler vowel ("uh") scores low."""
    m = an.ff.mfcc
    a = m[1:, an.ff.at(*burst)].mean(axis=1)
    b = m[1:, an.ff.at(word_start, word_start + dur)].mean(axis=1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else 0.0


# a local span flaw must exceed this multiple of the reference's own variability
MIN_Z = {"rushed": 0.8, "dragging": 0.8, "trailing_off": 1.0}
MIN_Z_DEFAULT = 1.5
SPAN_TYPES = {"rushed", "dragging", "trailing_off", "monotone", "mumbling"}
POINT_TYPES = {"filler", "stutter", "awkward_pause"}


def _resolve(regions: list[Region]) -> list[Region]:
    """Remove secondary detections explained by a stronger co-located flaw:
    muffled articulation (low-pass) also lowers measured loudness, so a mild
    volume drop inside a mumbling region is not reported twice."""
    out = []
    gate = {} if os.environ.get("SPEECHLENS_NO_GATE") else GATES
    for r in regions:
        g = gate.get(r.type)
        if g is not None and r.magnitude is not None and r.scope == "local":
            # data-fitted significance gate (tools/fit_thresholds.py): deviations
            # within natural take-to-take variation are not reported
            if (r.magnitude < g["min"]) if g["worse"] == "higher" else (r.magnitude > g["min"]):
                continue
        z = r.metrics[0].z if r.metrics else None
        if r.scope == "local" and z is not None and abs(z) < MIN_Z.get(r.type, MIN_Z_DEFAULT):
            continue
        overlap = lambda o: min(r.end, o.end) - max(r.start, o.start)
        inside = lambda kind: any(o.type == kind and o.start - 0.2 <= r.start and r.end <= o.end + 0.2 for o in regions)
        # rushing eats pauses: missing pauses inside a rushed span are the same flaw
        if r.type == "pause_omission" and inside("rushed"):
            continue
        # dragging stretches the natural pauses too: a longer pause where the
        # reference already paused, inside a dragging span, is the same flaw
        if r.type == "awkward_pause" and inside("dragging") and r.metrics[0].reference >= 0.2 \
                and r.metrics[0].participant <= 2.6 * r.metrics[0].reference:
            continue
        # a short span deviation around a point disfluency is the disfluency:
        # the aligner folds an inserted "uh", pause or repeated onset into the
        # neighbouring words, which then look slow, quiet or flat
        if r.type in SPAN_TYPES and r.scope == "local" and (r.word_end - r.word_start) <= 5 and any(
                o.type in POINT_TYPES and overlap(o) > -0.15 for o in regions):
            continue
        # a fade-out shrinks energy-refined word boundaries, which reads as rushing
        if r.type == "rushed" and r.severity <= 3 and any(
                o.type in ("trailing_off", "mumbling") and overlap(o) > 0.5 * (r.end - r.start) for o in regions):
            continue
        if r.type == "trailing_off" and r.severity <= 2:
            if any(o.type == "mumbling" and min(r.end, o.end) - max(r.start, o.start) > 0.5 * (r.end - r.start)
                   for o in regions):
                continue
        out.append(r)
    return out


class Sev(int):
    """A 1..5 severity that remembers the raw magnitude it was derived from
    (used to fit severity tables from data, see tools/calibrate.py)."""
    def __new__(cls, level, magnitude):
        obj = super().__new__(cls, int(np.clip(level, 1, 5)))
        obj.magnitude = float(magnitude)
        return obj


import os

CALIBRATION = {}
GATES = {}
_cal_path = __import__("pathlib").Path(__file__).with_name("calibration.json")
if _cal_path.exists():
    import json as _json
    CALIBRATION = _json.loads(_cal_path.read_text())
    GATES = CALIBRATION.pop("_gates", {})


def _sevt(ftype, value, table, increasing=True):
    """Severity for a flaw type: the data-fitted table when one exists, else
    the injection engine's own parameter table."""
    if ftype in CALIBRATION:
        table, increasing = CALIBRATION[ftype]["table"], CALIBRATION[ftype]["increasing"]
    return Sev(_sev(value, table, increasing), value)


def _sev(value, table, increasing=True):
    """Map a magnitude to severity 1..5 using the midpoints of `table`."""
    t = np.asarray(table, float)
    mids = (t[1:] + t[:-1]) / 2
    if increasing:
        return int(1 + np.sum(value > mids))
    return int(1 + np.sum(value < mids))


def _conf(z, z0=2.0):
    return float(np.clip(1 - np.exp(-max(0.0, abs(z) - 0.5) / z0), 0.05, 0.99))


# ------------------------------------------------------------------- main ---
def compare(ref: Analysis, par: Analysis) -> dict:
    pairs = pair_words(ref, par)
    if len(pairs) < 5:
        raise ValueError("fewer than 5 words could be matched between the two readings")
    R = [ref.wf[i] for i, _ in pairs]
    Pw = [par.wf[j] for _, j in pairs]
    n = len(pairs)
    regions: list[Region] = []
    glob: dict = {}

    def P(j):  # participant word index of pair j
        return pairs[j][1]

    def mk(t, j0, j1, sev, conf, metrics, scope="local"):
        a, b = P(j0), P(j1)
        reg = Region(t, par.words[a].start, par.words[b].end, a, b, sev, round(conf, 3),
                     metrics, scope, ref.words[pairs[j0][0]].start, ref.words[pairs[j1][0]].end,
                     " ".join(w.text for w in par.words[a:b + 1]))
        regions.append(reg)
        return reg

    # ---------------- pace (articulation rate, pauses excluded) ----------
    syl = np.array([w.syllables for w in R], float)
    dr = np.array([w.dur for w in R])
    dp = np.array([w.dur for w in Pw])
    rr = np.array([_window(syl, dr, j, 1) for j in range(n)])
    rp = np.array([_window(syl, dp, j, 1) for j in range(n)])
    s = np.log2(rp / rr)
    g = float(np.nanmedian(s))
    loc = _smooth(s - g, 3)
    spread = _robust_spread(np.log2(rr / np.nanmedian(rr)), 0.08)
    glob["pace"] = {"log2_ratio": round(g, 3), "reference_sps": round(float(np.nansum(syl) / np.nansum(dr)), 2),
                    "participant_sps": round(float(np.nansum(syl) / np.nansum(dp)), 2)}
    word_ratio = np.log2(dr / np.maximum(dp, 1e-3)) - g  # per word, + = faster than reference
    for sign, t in ((1, "rushed"), (-1, "dragging")):
        for j0, j1 in _runs(sign * loc > TH["pace_log2"], min_len=3, hole=1):
            # a real pace change is spread over the words; one word inflated by
            # an absorbed filler or pause only moves the 3-word windows around it
            wr = sign * word_ratio[j0:j1 + 1]
            if np.median(wr) < 0.5 * TH["pace_log2"] or np.mean(wr > 0.1) < 0.6:
                continue
            m = float(np.nanmean(loc[j0:j1 + 1]))
            ref_r = float(syl[j0:j1 + 1].sum() / dr[j0:j1 + 1].sum())
            par_r = float(syl[j0:j1 + 1].sum() / dp[j0:j1 + 1].sum())
            factor = 2 ** (-m)  # duration factor relative to reference
            tbl = [0.88, 0.80, 0.72, 0.64, 0.55] if t == "rushed" else [1.12, 1.24, 1.36, 1.48, 1.62]
            sev = _sevt(t, factor, tbl, increasing=(t == "dragging"))
            mk(t, j0, j1, sev, _conf(m / spread), [
                Metric("articulation_rate", "syll/s", ref_r, par_r, par_r - ref_r, m / spread,
                       f"|log2 ratio| > {TH['pace_log2']} (local, after removing global offset {g:+.2f})")])
    if abs(g) > TH_GLOBAL["pace_log2"]:
        t = "rushed" if g > 0 else "dragging"
        mk(t, 0, n - 1, _sevt(("rushed" if g > 0 else "dragging"), 2 ** -g, [0.88, 0.80, 0.72, 0.64, 0.55] if g > 0 else [1.12, 1.24, 1.36, 1.48, 1.62],
                             increasing=(g < 0)), _conf(g / spread),
           [Metric("overall_articulation_rate", "syll/s", glob["pace"]["reference_sps"], glob["pace"]["participant_sps"],
                   glob["pace"]["participant_sps"] - glob["pace"]["reference_sps"], g / spread,
                   f"|global log2 ratio| > {TH_GLOBAL['pace_log2']}")], scope="global")

    # ---------------- pitch variety (F0 spread over 5-word windows) ------
    sr_ = np.array([_f0_spread(ref, [pairs[k][0] for k in range(max(0, j - 2), min(n, j + 3))]) for j in range(n)])
    sp_ = np.array([_f0_spread(par, [pairs[k][1] for k in range(max(0, j - 2), min(n, j + 3))]) for j in range(n)])
    s = np.log2(np.maximum(sp_, 0.2) / np.maximum(sr_, 0.2))
    g = float(np.nanmedian(s))
    loc = s - g
    spread = _robust_spread(np.log2(np.maximum(sr_, 0.2) / np.nanmedian(sr_)), 0.12)
    glob["pitch"] = {"log2_ratio": round(g, 3), "reference_range_st": round(_f0_spread(ref, [i for i, _ in pairs]), 2),
                     "participant_range_st": round(_f0_spread(par, [j for _, j in pairs]), 2)}
    for j0, j1 in _runs(loc < TH["pitch_log2"], min_len=4, hole=1):
        # trim the 2-word window bleed at each end
        if j1 - j0 >= 6:
            j0, j1 = j0 + 1, j1 - 1
        m = float(np.nanmean(loc[j0:j1 + 1]))
        keep = 2 ** m
        rng_r = _f0_spread(ref, [pairs[k][0] for k in range(j0, j1 + 1)])
        rng_p = _f0_spread(par, [pairs[k][1] for k in range(j0, j1 + 1)])
        mk("monotone", j0, j1, _sevt("monotone", 1 - keep, [0.35, 0.55, 0.70, 0.85, 0.97]), _conf(m / spread),
           [Metric("pitch_range_p10_p90", "semitones", rng_r, rng_p, rng_p - rng_r, m / spread,
                   f"log2(spread ratio) < {TH['pitch_log2']}")])
    if g < TH_GLOBAL["pitch_log2"]:
        mk("monotone", 0, n - 1, _sevt("monotone", 1 - 2 ** g, [0.35, 0.55, 0.70, 0.85, 0.97]), _conf(g / spread),
           [Metric("overall_pitch_range", "semitones", glob["pitch"]["reference_range_st"],
                   glob["pitch"]["participant_range_st"], glob["pitch"]["participant_range_st"] - glob["pitch"]["reference_range_st"],
                   g / spread, f"global log2 ratio < {TH_GLOBAL['pitch_log2']}")], scope="global")

    # ---------------- loudness (speaker-relative dB) ---------------------
    ir = np.array([w.int_mean_db for w in R])
    ip = np.array([w.int_mean_db for w in Pw])
    s = ip - ir
    g = float(np.nanmedian(s))
    loc = _smooth(s - g, 3)
    spread = _robust_spread(ir - np.nanmedian(ir), 1.5)
    glob["volume"] = {"offset_db": round(g, 2)}
    for j0, j1 in _runs(loc < TH["vol_db"], min_len=3, hole=1):
        m = float(np.nanmean(loc[j0:j1 + 1]))
        worst = float(np.nanmin(loc[j0:j1 + 1]))
        mk("trailing_off", j0, j1, _sevt("trailing_off", -worst, [4, 7, 10, 14, 18]), _conf(m / spread),
           [Metric("loudness_vs_reference", "dB", float(np.nanmean(ir[j0:j1 + 1])), float(np.nanmean(ip[j0:j1 + 1])), m, m / spread,
                   f"speaker-relative level drop > {-TH['vol_db']} dB"),
            Metric("deepest_drop", "dB", 0.0, worst, worst)])

    # ---------------- articulation clarity (high-band share) ------------
    hr = np.array([w.hiband for w in R])
    hp = np.array([w.hiband for w in Pw])
    hr4 = np.array([w.hiband4 for w in R])
    hp4 = np.array([w.hiband4 for w in Pw])
    # mean of the >2 kHz and >4 kHz log ratios: the 4 kHz band catches mild
    # muffling that leaves 2-4 kHz intact
    s = 0.5 * (np.log2(np.maximum(hp, 1e-4) / np.maximum(hr, 1e-4)) +
               np.log2(np.maximum(hp4, 1e-5) / np.maximum(hr4, 1e-5)))
    g = float(np.nanmedian(s))
    loc = _smooth(s - g, 3)
    spread = _robust_spread(0.5 * (np.log2(np.maximum(hr, 1e-4) / np.nanmedian(hr)) +
                                   np.log2(np.maximum(hr4, 1e-5) / np.nanmedian(hr4))), 0.25)
    cr = np.array([w.conf for w in R])
    cp = np.array([w.conf for w in Pw])
    glob["clarity"] = {"log2_ratio": round(g, 3)}
    for j0, j1 in _runs(loc < TH["clar_log2"], min_len=3, hole=1):
        m = float(np.nanmean(loc[j0:j1 + 1]))
        mets = [Metric("energy_above_2kHz", "%", float(np.nanmean(hr[j0:j1 + 1])) * 100, float(np.nanmean(hp[j0:j1 + 1])) * 100,
                       float(np.nanmean(hp[j0:j1 + 1]) - np.nanmean(hr[j0:j1 + 1])) * 100, m / spread,
                       f"mean log2(high-band ratio, >2k & >4k) < {TH['clar_log2']}"),
                Metric("energy_above_4kHz", "%", float(np.nanmean(hr4[j0:j1 + 1])) * 100, float(np.nanmean(hp4[j0:j1 + 1])) * 100,
                       float(np.nanmean(hp4[j0:j1 + 1]) - np.nanmean(hr4[j0:j1 + 1])) * 100)]
        if cr.any():
            mets.append(Metric("recognizer_confidence", "", float(cr[j0:j1 + 1].mean()), float(cp[j0:j1 + 1].mean()),
                               float(cp[j0:j1 + 1].mean() - cr[j0:j1 + 1].mean())))
        mk("mumbling", j0, j1, _sevt("mumbling", -m, [0.8, 1.6, 2.6, 3.6, 5.0]), _conf(m / spread), mets)

    # ---------------- pauses & disfluencies (per word boundary) ----------
    # Each boundary is examined over the zone [end of word j, end of word j+1]
    # in both readings. Disfluencies always ADD time there, which separates
    # them from the small boundary shifts two independent alignments produce.
    st_p, st_r = _steady_mask(par), _steady_mask(ref)
    pause_flags = np.zeros(n, bool)
    pause_info = {}
    # time added at each boundary, and the typical amount added at nearby
    # boundaries: a disfluency adds time at ONE spot, a slow passage everywhere
    raw_extra = np.full(n, np.nan)
    pause_extra = np.full(n, np.nan)
    for j in range(n - 1):
        if pairs[j + 1][0] == pairs[j][0] + 1 and pairs[j + 1][1] == pairs[j][1] + 1:
            raw_extra[j] = ((par.words[P(j + 1)].end - par.words[P(j)].start)
                            - (ref.words[pairs[j + 1][0]].end - ref.words[pairs[j][0]].start))
            pause_extra[j] = Pw[j].pause_after - R[j].pause_after

    def local_base(x, j, h=3):
        w = np.r_[x[max(0, j - h):j], x[j + 1:j + 1 + h]]
        w = w[np.isfinite(w)]
        return float(np.percentile(w, 75)) if len(w) else 0.0

    last_filler_end = -1.0
    for j in range(n - 1):
        if not np.isfinite(raw_extra[j]):
            continue
        a, b = par.words[P(j)], par.words[P(j + 1)]
        ra, rb = ref.words[pairs[j][0]], ref.words[pairs[j + 1][0]]
        pb, pp = R[j].pause_after, Pw[j].pause_after
        extra = raw_extra[j] - max(0.0, local_base(raw_extra, j))   # added here beyond the local trend

        # repeated onset: short fragments before the completed word, the first
        # one spectrally matching the word's own onset
        if extra >= 0.12:
            fr_p = _bursts(par, a.end + 0.03, b.end)
            fr_r = _bursts(ref, ra.end + 0.03, rb.end)
            if len(fr_p) >= 2 and len(fr_p) > len(fr_r) and all(f[1] - f[0] < 0.22 for f in fr_p[:-1]):
                sim = _onset_similarity(par, fr_p[0], fr_p[-1][0])
                rep = len(fr_p) - max(1, len(fr_r))
                if sim > (0.88 if rep <= 1 else 0.80):
                    regions.append(Region("stutter", round(fr_p[0][0], 3), b.end, P(j + 1), P(j + 1),
                                          _sevt("stutter", rep, [1, 1.5, 2, 2.5, 3]), round(min(0.95, 0.4 + 0.5 * (sim - 0.8) / 0.2 + 0.1 * rep), 3),
                                          [Metric("onset_repetitions", "count", 0, rep, rep, None,
                                                  "extra fragment(s) before the word, +time at this boundary"),
                                           Metric("fragment_vs_word_onset_similarity", "cosine", 0, sim, sim, None,
                                                  "MFCC cosine > 0.80 = the same onset repeated"),
                                           Metric("time_added", "s", 0, extra, extra)],
                                          text=b.text))
                    continue

        # filled pause: a held, steady vowel the reference does not have here,
        # whether it sits in the gap or the aligner folded it into a word
        if extra >= 0.15:
            lr_, _, _ = _longest_run(st_r, ra.start + 0.05, rb.end)
            lp_, sp_, ep_ = _run_near(st_p, a.start + 0.05, b.end, 0.5 * (a.end + b.start), lr_ + 0.14)
            gap_voiced = Pw[j].filled_after - R[j].filled_after
            if ((lp_ - lr_ >= 0.14) or gap_voiced >= TH["filled_s"]) and \
                    (sp_ if lp_ - lr_ >= 0.14 else a.end) >= last_filler_end - 0.05:
                held = max(lp_ - lr_, gap_voiced)
                s0, e0 = (sp_, ep_) if lp_ - lr_ >= 0.14 else (a.end, b.start)
                last_filler_end = e0
                regions.append(Region("filler", round(s0, 3), round(e0, 3), P(j), P(j + 1),
                                      _sevt("filler", held + 0.1, [0.28, 0.36, 0.45, 0.55, 0.65]), _conf(held / 0.06),
                                      [Metric("held_steady_voicing", "s", lr_, lp_, lp_ - lr_, None,
                                              "steady pitch + unchanging spectrum >= 140 ms longer than reference"),
                                       Metric("time_added", "s", 0, extra, extra)],
                                      text=f"{a.text} … {b.text}"))
                continue

        pe = pause_extra[j] - max(0.0, local_base(pause_extra, j))
        # at punctuation a longer pause is usually a deliberate, dramatic one;
        # only an extreme one counts there. Mid-phrase, the normal threshold.
        need = TH["pause_extra_s"] if not R[j].punct_after else TH["pause_extra_punct_s"]
        if pe >= need and pp > 1.8 * pb + 0.15:
            regions.append(Region("awkward_pause", round(Pw[j].end, 3), round(b.start, 3), P(j), P(j + 1),
                                  _sevt("awkward_pause", pp - pb, [0.35, 0.60, 0.90, 1.30, 1.80]), _conf((pp - pb) / 0.15),
                                  [Metric("pause_length", "s", pb, pp, pp - pb, None,
                                          f">= {TH['pause_extra_s']} s longer than reference" +
                                          ("" if R[j].punct_after else ", mid-phrase (no punctuation)"))],
                                  text=f"{a.text} ▮ {b.text}"))
            continue
        if pb >= 0.18 and pp < TH["pause_keep"] * pb and pb - pp >= 0.12:
            pause_flags[j] = True
            pause_info[j] = (pb, pp)
    # merge omitted pauses that sit within 8 words of each other; a lone one
    # must be a substantial pause to count
    idx = np.nonzero(pause_flags)[0]
    groups = []
    for j in idx:
        if groups and j - groups[-1][-1] <= 8:
            groups[-1].append(j)
        else:
            groups.append([j])
    for grp in groups:
        pbs = np.array([pause_info[j][0] for j in grp])
        pps = np.array([pause_info[j][1] for j in grp])
        if len(grp) < 2:  # running through a passage, not one shortened pause
            continue
        keep = float((pps / pbs).mean())
        j0, j1 = grp[0], min(n - 1, grp[-1] + 1)
        mk("pause_omission", j0, j1, _sevt("pause_omission", keep, [0.60, 0.45, 0.30, 0.15, 0.04], increasing=False),
           min(0.95, 0.45 + 0.15 * len(grp)),
           [Metric("mean_pause_at_punctuation", "s", float(pbs.mean()), float(pps.mean()), float(pps.mean() - pbs.mean()), None,
                   f"reference pause >= 0.18 s kept at < {int(TH['pause_keep'] * 100)} %"),
            Metric("pauses_lost", "count", len(grp), 0, -len(grp))])

    regions = _resolve(regions)
    regions.sort(key=lambda r: (r.scope != "global", r.start))
    return {"pairs": len(pairs), "regions": regions, "global": glob,
            "coverage": round(len(pairs) / max(1, len(par.words)), 3)}
