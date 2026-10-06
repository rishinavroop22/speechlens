"""Build a blind listening audit of findings on the human recordings.

The detector's precision on human takes is measured against the scripted
labels only, so an unscripted but real slip (a lost pause, a hesitation)
counts as a false positive. To estimate real precision, a listener rates
30 short snippets, each heard against the same words in the speaker's clean
take, without knowing why the snippet was picked:
  * 16 findings that match no scripted label   ("extra")
  * 7 findings that match a scripted label     ("scripted", rater sensitivity)
  * 7 random spans with no finding and no label ("null", rater base rate)
Writes the page to build/audit/ and the hidden key to results/audit_key.json.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import human_view, match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data/dataset"
OUT = ROOT / "build/audit"


def cut(src, a, b, dst):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(src),
                    "-af", "afade=t=in:d=0.05,areverse,afade=t=in:d=0.08,areverse", "-ac", "1",
                    "-c:a", "libmp3lame", "-b:a", "48k", str(dst)], check=True)


def words_of(cid):
    return json.loads((DS / "labels" / f"{cid}.json").read_text())["words"]


def main():
    rng = random.Random(20261007)
    man = {c["id"]: c for c in json.loads((DS / "manifest.json").read_text())["clips"]}
    preds = {r["id"]: r for r in json.loads((ROOT / "results/predictions.json").read_text())}
    extra, scripted, null = [], [], []
    for cid, c in man.items():
        if c["kind"] != "human":
            continue
        local = [p for p in preds[cid]["preds"] if p["scope"] == "local"]
        labels, local = human_view(c, local)
        m = match(labels, local, 0.3)
        used = {j for _, j, _ in m}
        W = words_of(cid)

        def span_words(a, b):
            idx = [w["i"] for w in W if w["end"] > a and w["start"] < b]
            return (min(idx), max(idx)) if idx else None
        for j, p in enumerate(local):
            sw = span_words(p["start"], p["end"])
            if sw:
                (scripted if j in used else extra).append({"clip": cid, "type": p["type"], "w": sw})
        # null spans: 4-6 words away from any label or finding
        busy = set()
        for x in list(labels) + local:
            sw = span_words(x["start"] - 0.5, x["end"] + 0.5)
            if sw:
                busy |= set(range(sw[0], sw[1] + 1))
        for _ in range(20):
            a = rng.randrange(1, len(W) - 7)
            b = a + rng.randrange(3, 6)
            if not busy & set(range(a, b + 1)):
                null.append({"clip": cid, "type": None, "w": (a, b)})
                break
    pick = (rng.sample(extra, min(16, len(extra))) + rng.sample(scripted, min(7, len(scripted)))
            + rng.sample(null, min(7, len(null))))
    groups = ["extra"] * min(16, len(extra)) + ["scripted"] * min(7, len(scripted)) + ["null"] * min(7, len(null))
    order = list(range(len(pick)))
    rng.shuffle(order)
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    items, key = [], {}
    for n, k in enumerate(order, start=1):
        it, g = pick[k], groups[k]
        c = man[it["clip"]]
        a, b = it["w"]
        Wp, Wr = words_of(it["clip"]), words_of(c["reference_clip"])
        pa, pb = max(0, Wp[a]["start"] - 0.6), Wp[b]["end"] + 0.6
        ra, rb = max(0, Wr[a]["start"] - 0.6), Wr[b]["end"] + 0.6
        if pb - pa > 9:
            pb = pa + 9
        iid = f"q{n:02d}"
        cut(DS / "clips" / f"{it['clip']}.wav", pa, pb, OUT / "clips" / f"{iid}_p.mp3")
        cut(DS / "clips" / f"{c['reference_clip']}.wav", ra, rb, OUT / "clips" / f"{iid}_r.mp3")
        text = " ".join(w["text"] for w in Wp[max(0, a - 1): b + 2])
        items.append({"id": iid, "text": text})
        key[iid] = {"group": g, "claimed": it["type"], "clip": it["clip"], "words": [a, b]}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/audit_key.json").write_text(json.dumps(key, indent=1))
    (OUT / "items.json").write_text(json.dumps(items))
    print(len(items), "items;", {g: groups.count(g) for g in set(groups)}, "pool sizes", len(extra), len(scripted), len(null))


if __name__ == "__main__":
    main()
