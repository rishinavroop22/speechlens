"""Transcript normalization: display tokens are kept verbatim, while each token
also gets a normalized spelling (A-Z and apostrophe) for the acoustic model."""
from __future__ import annotations

import re
from dataclasses import dataclass

from num2words import num2words

_PUNCT_PAUSE = {",": "comma", ";": "clause", ":": "clause", ".": "sentence", "!": "sentence",
                "?": "sentence", "—": "clause", "–": "clause"}


@dataclass
class Token:
    text: str            # as written, e.g. "country;"
    norm: str            # model spelling, e.g. "COUNTRY"
    punct_after: str | None  # "comma" | "clause" | "sentence" | None


_MONTHS = {"JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER",
           "OCTOBER", "NOVEMBER", "DECEMBER"}


def _spell(word: str) -> str:
    w = word.replace("’", "'")
    if re.fullmatch(r"\d+(st|nd|rd|th)", w, re.I):
        w = num2words(int(re.sub(r"\D", "", w)), to="ordinal")
    elif re.fullmatch(r"\d{4}", w) and 1100 <= int(w) <= 2099:
        w = num2words(int(w), to="year")
    elif re.fullmatch(r"[\d,]+(\.\d+)?", w):
        w = num2words(float(w.replace(",", "")) if "." in w else int(w.replace(",", "")))
    w = w.replace("%", " percent").replace("&", " and ")
    return re.sub(r"[^A-Z' ]", " ", w.upper()).strip()


def tokenize(transcript: str) -> list[Token]:
    out: list[Token] = []
    for raw in re.findall(r"\S+", re.sub(r"\s*(--|—|–)\s*", "— ", transcript)):
        trail = re.search(r"[,;:.!?—–]+[\"'”’)]*$", raw)
        punct = None
        if trail:
            marks = [c for c in trail.group(0) if c in _PUNCT_PAUSE]
            if marks:
                kinds = [_PUNCT_PAUSE[c] for c in marks]
                punct = "sentence" if "sentence" in kinds else ("clause" if "clause" in kinds else "comma")
        core = raw.strip("\"'“”‘’()[]—–,;:.!?")
        if out and out[-1].norm in _MONTHS and re.fullmatch(r"\d{1,2}", core) and 1 <= int(core) <= 31:
            core = core + "th"  # "December 7" is spoken "December seventh"
        norm = _spell(core)
        if not norm:
            if out and punct:  # stray dash etc. attaches to previous word
                out[-1].punct_after = punct
            continue
        # a number can expand to several spoken words; keep them as one display token
        out.append(Token(text=raw, norm=norm.replace(" ", "|"), punct_after=punct))
    return out


def count_syllables(norm: str) -> int:
    """Vowel-group heuristic on the normalized spelling (good to ~±1 per word)."""
    total = 0
    for w in norm.replace("'", "").split("|"):
        w = w.lower()
        if not w:
            continue
        groups = re.findall(r"[aeiouy]+", w)
        n = len(groups)
        if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
            n -= 1
        if w.endswith("ed") and len(w) > 3 and w[-3] not in "td" and n > 1:
            n -= 1
        total += max(1, n)
    return total
