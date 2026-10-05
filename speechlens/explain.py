"""Deterministic causal explanations and rubric scoring.

Every sentence is filled from measured values - no language model is
involved - so the same audio always yields the same words and numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contrast import Region

SEV_WORD = {1: "slight", 2: "mild", 3: "moderate", 4: "marked", 5: "severe"}

WHY = {
    "rushed": "Faster articulation compresses the stressed syllables and the micro-pauses listeners use to "
              "segment phrases, so key words lose emphasis and comprehension drops.",
    "dragging": "Stretched words without matching emphasis read as hesitation and dilute the rhythm of the "
                "phrase; the audience's attention drifts between content words.",
    "monotone": "Pitch movement is the main cue English listeners use to find the focus word of a phrase; "
                "with a flattened contour every word carries equal weight and the line sounds recited.",
    "trailing_off": "The loudness falls away across the phrase, so its final words - often the payoff - "
                    "drop toward the noise floor of the room.",
    "mumbling": "Energy above 2 kHz carries consonant bursts and fricatives (t, k, s, f); losing it blurs "
                "word boundaries and makes the speech sound muffled even at normal volume.",
    "pause_omission": "Pauses at punctuation mark the end of an idea and give the audience time to absorb it; "
                      "running through them merges separate thoughts into one stream.",
    "awkward_pause": "A long silence where the reference has none - especially mid-phrase - breaks the "
                     "syntactic unit and signals lost place rather than deliberate emphasis.",
    "filler": "Voiced sound inside a pause that maps to no word in the transcript is a filled pause "
              "(\"uh\"/\"um\"), heard as hesitation and penalised under fluency.",
    "stutter": "Repeated onsets before the word is completed fragment it and interrupt the speech rhythm.",
}

ADVICE = {
    "rushed": "Slow to the reference pace here and let the pauses between clauses breathe.",
    "dragging": "Tighten these words to the reference pace; keep length only on the stressed word.",
    "monotone": "Lift pitch on the focus word of each clause and let it fall at the end of the thought.",
    "trailing_off": "Support the breath through the end of the phrase; the last words need the same energy as the first.",
    "mumbling": "Open the mouth more and finish final consonants crisply.",
    "pause_omission": "Hold a clear pause at each punctuation mark, about as long as the reference.",
    "awkward_pause": "Keep this phrase in one breath; pause at the punctuation instead.",
    "filler": "Replace the filler with silence - a clean pause sounds deliberate.",
    "stutter": "Begin the word once, deliberately; a short breath before it helps.",
}


def _fmt(v, unit):
    if unit == "%":
        return f"{v:.1f}%"
    if unit == "s":
        return f"{v * 1000:.0f} ms" if abs(v) < 1 else f"{v:.2f} s"
    if unit in ("dB", "semitones"):
        return f"{v:.1f} {'st' if unit == 'semitones' else 'dB'}"
    if unit == "syll/s":
        return f"{v:.2f} syll/s"
    if unit == "count":
        return f"{int(round(v))}"
    return f"{v:.2f}"


def explain(r: Region) -> str:
    m = r.metrics[0]
    where = (f"Across the whole reading" if r.scope == "global"
             else f"From {r.start:.2f}s to {r.end:.2f}s (“{_clip(r.text)}”)")
    ref, par = _fmt(m.reference, m.unit), _fmt(m.participant, m.unit)
    rel = ""
    if m.unit in ("syll/s", "semitones", "%", "s") and m.reference:
        rel = f" ({(m.participant / m.reference - 1) * 100:+.0f}%)"
    z = f"; that is {abs(m.z):.1f}× the reference speaker's natural word-to-word variation" if m.z is not None else ""
    LQ, RQ = "\u201c", "\u201d"
    pair_txt = r.text.replace(" \u25ae ", f"{RQ} and {LQ}").replace(" \u2026 ", f"{RQ} and {LQ}")
    head = {
        "rushed": f"{where}, articulation rate was {par} against {ref} in the reference{rel}{z}.",
        "dragging": f"{where}, articulation rate slowed to {par} against {ref} in the reference{rel}{z}.",
        "monotone": f"{where}, pitch range (10th–90th percentile) narrowed to {par} from {ref}{rel}{z}.",
        "trailing_off": f"{where}, loudness relative to the speaker's own level fell {abs(m.delta):.1f} dB below the reference"
                        + (f", bottoming out {abs(r.metrics[1].participant):.1f} dB down" if len(r.metrics) > 1 else "") + f"{z}.",
        "mumbling": f"{where}, the share of energy above 2 kHz dropped to {par} from {ref}{rel}{z}.",
        "pause_omission": f"{where}, {int(r.metrics[1].reference) if len(r.metrics) > 1 else 'several'} pause(s) at punctuation "
                          f"shrank from {ref} to {par} on average.",
        "awkward_pause": f"At {r.start:.2f}s, between {LQ}{pair_txt}{RQ}, a {par} silence "
                         f"replaced the reference's {ref} pause.",
        "filler": f"At {r.start:.2f}s, {par} of voiced sound appears in the pause between "
                  f"{LQ}{pair_txt}{RQ} where the reference has {ref}.",
        "stutter": f"At {r.start:.2f}s, the onset of “{r.text}” is repeated {par} time(s) before the word is completed"
                   + (f" (burst-to-onset spectral similarity {r.metrics[1].participant:.2f})" if len(r.metrics) > 1 else "") + ".",
    }[r.type]
    return f"{head} {WHY[r.type]} Fix: {ADVICE[r.type]}"


def _clip(t, n=60):
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + " …"


# ---------------------------------------------------------------- rubric ---
RUBRIC_DIR = Path(__file__).resolve().parent / "rubrics"
DIMENSIONS = ["pace", "pausing", "fluency", "pitch_variety", "volume", "clarity"]
DIM_LABEL = {"pace": "Pace", "pausing": "Pausing", "fluency": "Fluency", "pitch_variety": "Pitch variety",
             "volume": "Volume control", "clarity": "Clarity"}


def load_rubric(name: str = "declamation") -> dict:
    return json.loads((RUBRIC_DIR / f"{name}.json").read_text())


def list_rubrics() -> list[dict]:
    return [json.loads(p.read_text()) | {"id": p.stem} for p in sorted(RUBRIC_DIR.glob("*.json"))]


def score(regions: list[Region], n_words: int, duration: float, rubric: dict) -> dict:
    """Each dimension starts at 10. A region costs severity x weight, where span
    flaws scale with the share of the speech they cover and point flaws
    (pauses, fillers, stutters) cost a fixed amount each. The overall score is
    the rubric-weighted mean, x10 -> 0..100."""
    pen = {d: 0.0 for d in DIMENSIONS}
    for r in regions:
        if r.scope == "global":
            pen[r.dimension] += r.severity * 1.6
        elif r.type in ("awkward_pause", "filler", "stutter"):
            pen[r.dimension] += r.severity * 0.45
        else:
            cover = (r.word_end - r.word_start + 1) / max(1, n_words)
            pen[r.dimension] += r.severity * (0.55 + 4.0 * cover)
    dims = {d: round(max(0.0, 10.0 - pen[d]), 2) for d in DIMENSIONS}
    w = rubric["weights"]
    total = sum(w[d] for d in DIMENSIONS)
    overall = round(10 * sum(dims[d] * w[d] for d in DIMENSIONS) / total, 1)
    return {"overall": overall, "dimensions": dims, "rubric": rubric.get("name"),
            "band": next(b["label"] for b in rubric["bands"] if overall >= b["min"])}
