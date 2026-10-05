"""Generate the human recording protocol and its machine-readable script.

Each recorder reads the same excerpt four times:
  A  clean      - their best delivery (no flaws)
  B  pace       - rush one sentence, drag another, add a long pause mid-phrase
  C  voice      - monotone one sentence, trail off one, mumble one
  D  fluency    - "um"/"uh" at two marked spots, repeat one word onset, run
                  through the pauses of one sentence
Because the flaw locations are fixed by the script, human recordings get
labels automatically: we align each take and turn the scripted word spans
into time spans (tools/label_human.py).

Writes data/human/script.json and docs/recording_sheet.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens.text import tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "data/library"
SPEECHES = ["reagan", "jfk", "obama"]


def sentences(text):
    toks = tokenize(text)
    out, cur = [], []
    for i, t in enumerate(toks):
        cur.append(i)
        if t.punct_after == "sentence":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return toks, [s for s in out if len(s) >= 5]


def pick_mid(toks, sent, avoid_punct=True):
    """A word boundary in the middle of a sentence, not at punctuation."""
    mid = sent[len(sent) // 2]
    for d in range(len(sent)):
        for k in (mid + d, mid - d):
            if sent[1] <= k < sent[-2] and (not avoid_punct or not toks[k].punct_after):
                return k
    return mid


def stutter_word(toks, sent):
    for k in sent[2:-1]:
        if re.match(r"[BCDGKPT]", toks[k].norm) and len(toks[k].norm) >= 4:
            return k
    return sent[len(sent) // 2]


def script_for(sid):
    text = (LIB / sid / "transcript.txt").read_text().strip()
    toks, S = sentences(text)
    n = len(S)
    pick = lambda f: S[min(n - 1, int(f * n))]
    plan = {"A": []}
    s1, s2, s3 = pick(0.15), pick(0.45), pick(0.75)
    k = pick_mid(toks, s2)
    plan["B"] = [
        {"type": "rushed", "words": [s1[0], s1[-1]], "say": "Read this sentence about twice as fast as normal, no pauses."},
        {"type": "awkward_pause", "words": [k, k + 1], "say": f"Stop for about two seconds after “{toks[k].text}”, then carry on."},
        {"type": "dragging", "words": [s3[0], s3[-1]], "say": "Read this sentence very slowly, stretching every word."},
    ]
    s1, s2, s3 = pick(0.05), pick(0.4), pick(0.7)
    plan["C"] = [
        {"type": "monotone", "words": [s1[0], s1[-1]], "say": "Read this sentence on one flat note, like a robot."},
        {"type": "trailing_off", "words": [s2[len(s2) // 3], s2[-1]], "say": "From the marked word, fade almost to a whisper by the end of the sentence."},
        {"type": "mumbling", "words": [s3[0], s3[-1]], "say": "Mumble this sentence: barely open your mouth, swallow the consonants."},
    ]
    s1, s2, s3 = pick(0.2), pick(0.5), pick(0.8)
    k1, k2 = pick_mid(toks, s1), pick_mid(toks, s3)
    st = stutter_word(toks, s2)
    with_punct = max(S, key=lambda s: sum(1 for i in s[:-1] if toks[i].punct_after))
    plan["D"] = [
        {"type": "filler", "words": [k1, k1 + 1], "say": f"Say a long “ummm” after “{toks[k1].text}”."},
        {"type": "stutter", "words": [st, st], "say": f"Stutter the start of “{toks[st].text}” two or three times (“{toks[st].norm[:2].lower()}-{toks[st].norm[:2].lower()}-{toks[st].text.lower().strip(',.;:')}”)."},
        {"type": "filler", "words": [k2, k2 + 1], "say": f"Say “uhh” after “{toks[k2].text}”."},
    ]
    if with_punct not in (s1, s2, s3):
        plan["D"].append({"type": "pause_omission", "words": [with_punct[0], with_punct[-1]],
                          "say": "Run straight through every comma in this sentence without pausing."})
    return {"id": sid, "text": text, "tokens": [t.text for t in toks], "takes": plan}


def render(scripts):
    meta = {d.name: json.loads((d / "meta.json").read_text()) for d in LIB.iterdir() if (d / "meta.json").exists()}
    out = ["# SpeechLens recording sheet", "",
           "Thank you for recording. Each person reads three short passages, four times each (A, B, C, D). "
           "About 15 minutes in total.", "",
           "## Before you start", "",
           "- Use a phone voice-recorder app in a quiet room, phone about 20 cm from your mouth.",
           "- One file per take. Name each file `yourname_passage_take`, for example `nikhil_reagan_B.m4a`.",
           "- Take **A** is your best, natural, confident reading. Takes **B, C, D** add the flaws marked in the text.",
           "- Only the marked parts get a flaw. Read everything else normally.",
           "- If you make an accidental mistake, just restart that take.", ""]
    marks = {"rushed": "FAST", "dragging": "SLOW", "awkward_pause": "PAUSE", "monotone": "FLAT",
             "trailing_off": "FADE", "mumbling": "MUMBLE", "filler": "UM", "stutter": "STUTTER", "pause_omission": "NO PAUSES"}
    for sc in scripts:
        m = meta.get(sc["id"], {})
        out += [f"## Passage: {m.get('speaker', sc['id'])}, {m.get('title', '')} (`{sc['id']}`)", ""]
        for take, items in sc["takes"].items():
            out += [f"### Take {take}" + (": clean, your best delivery" if not items else ""), ""]
            if not items:
                out += ["Read the passage naturally and well.", "", "> " + " ".join(sc["tokens"]), ""]
                continue
            for it in items:
                a, b = it["words"]
                snippet = " ".join(sc["tokens"][a: b + 1]) if it["type"] not in ("awkward_pause", "filler") else \
                    " ".join(sc["tokens"][max(0, a - 3): b + 3])
                out.append(f"- **{marks[it['type']]}**: {it['say']}  \n  “{snippet}”")
            out += ["", "Full passage:", ""]
            text = list(sc["tokens"])
            for it in items:
                a, b = it["words"]
                tag = marks[it["type"]]
                if it["type"] in ("awkward_pause", "filler"):
                    text[a] = f"{text[a]} **[{tag}]**"
                elif it["type"] == "stutter":
                    text[a] = f"**[{tag}]** {text[a]}"
                else:
                    text[a] = f"**[{tag} →** {text[a]}"
                    text[b] = f"{text[b]} **←]**"
            out += ["> " + " ".join(text), ""]
    return "\n".join(out)


def main():
    scripts = [script_for(s) for s in SPEECHES if (LIB / s).exists()]
    (ROOT / "data/human").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/human/script.json").write_text(json.dumps(scripts, indent=1))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/recording_sheet.md").write_text(render(scripts))
    print("wrote data/human/script.json and docs/recording_sheet.md")


if __name__ == "__main__":
    main()
