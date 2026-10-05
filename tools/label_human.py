"""Label the team's scripted human recordings.

Put files in data/raw/human/ named  <name>_<passage>_<take>.<ext>
(e.g. nikhil_reagan_B.m4a). For each take we force-align the passage text
and turn the scripted word spans (data/human/script.json) into time spans.
Output: data/dataset/clips/human__*.wav, labels/*.json, data/dataset/human.json
(merged into the manifest on the next build, or immediately with --merge).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speechlens import audio, features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/human"
DS = ROOT / "data/dataset"
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm", ".flac", ".3gp", ".amr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="also add the clips to manifest.json now")
    args = ap.parse_args()
    scripts = {s["id"]: s for s in json.loads((ROOT / "data/human/script.json").read_text())}
    (DS / "clips").mkdir(parents=True, exist_ok=True)
    (DS / "labels").mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(RAW.iterdir()) if RAW.exists() else []:
        if f.suffix.lower() not in AUDIO_EXT:
            continue
        m = re.fullmatch(r"(.+?)[_ -]+([a-z]+)[_ -]+([abcd])", f.stem.lower())
        if not m or m.group(2) not in scripts:
            print(f"  ! skipped {f.name} (expected name_passage_take, passage one of {list(scripts)})")
            continue
        who, sid, take = m.group(1), m.group(2), m.group(3).upper()
        sc = scripts[sid]
        y = audio.peak_normalize(audio.load(f))
        an = features.analyze(y, sc["text"])
        W = an.words
        labels = []
        for it in sc["takes"][take]:
            a, b = it["words"]
            t = it["type"]
            if t in ("awkward_pause", "filler"):
                s, e = W[a].end, W[b].start
            elif t == "stutter":
                s, e = (W[a - 1].end if a else W[a].start), W[a].end
            else:
                s, e = W[a].start, W[b].end
            labels.append({"type": t, "severity": 3, "start": round(s, 3), "end": round(e, 3),
                           "word_start": a, "word_end": b, "params": {"scripted": it["say"]}})
        cid = f"human__{who}_{sid}_{take}"
        audio.save(DS / "clips" / f"{cid}.wav", y)
        conf = sum(w.conf for w in W) / max(1, len(W))
        rec = {"id": cid, "source": sid, "kind": "human" if labels else "control",
               "control": None if labels else "human_clean", "speaker": who, "take": take,
               "title": f"{who.title()}, take {take}", "duration": round(len(y) / audio.SR, 2),
               "labels": labels, "max_severity": 3 if labels else 0, "align_conf": round(conf, 3)}
        (DS / "labels" / f"{cid}.json").write_text(json.dumps(rec | {"words": [w.to_dict() for w in W]}, indent=1))
        out.append(rec)
        flag = "  (low alignment confidence - check the take)" if conf < 0.6 else ""
        print(f"  {cid}: {len(labels)} labels, align conf {conf:.2f}{flag}")
    (DS / "human.json").write_text(json.dumps(out, indent=1))
    if args.merge and (DS / "manifest.json").exists():
        man = json.loads((DS / "manifest.json").read_text())
        man["clips"] = [c for c in man["clips"] if not c["id"].startswith("human__")] + out
        (DS / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{len(out)} human clips labelled")


if __name__ == "__main__":
    main()
