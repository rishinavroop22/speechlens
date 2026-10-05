"""Cross-tabulate false positives and misses against nearby ground truth."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import match
ROOT = Path(__file__).resolve().parent.parent
P = {r["id"]: r for r in json.load(open(ROOT / "results/predictions.json"))}
M = {c["id"]: c for c in json.load(open(ROOT / "data/dataset/manifest.json"))["clips"]}
near, miss, sevs = collections.Counter(), collections.Counter(), collections.defaultdict(list)
for cid, c in M.items():
    local = [p for p in P[cid]["preds"] if p["scope"] == "local"]
    m = match(c["labels"], local, 0.3); used = {j for _, j, _ in m}; hit = {i for i, _, _ in m}
    for j, p in enumerate(local):
        if j in used: continue
        ov = [l["type"] for l in c["labels"] if min(l["end"], p["end"]) - max(l["start"], p["start"]) > -0.3]
        near[(p["type"], tuple(sorted(set(ov))) or ("none",))] += 1
        sevs[p["type"]].append(p["severity"])
    for i, l in enumerate(c["labels"]):
        if i in hit: continue
        ov = [p["type"] for p in local if min(l["end"], p["end"]) - max(l["start"], p["start"]) > -0.2]
        miss[(l["type"], tuple(sorted(set(ov))) or ("nothing",))] += 1
print("FALSE POSITIVES (pred, overlapping truth):")
for k, v in near.most_common(int(sys.argv[1]) if len(sys.argv) > 1 else 15): print(" ", k, v)
print("FP severity histogram:", {t: collections.Counter(s) for t, s in sevs.items()})
print("MISSES (truth, predicted there):")
for k, v in miss.most_common(int(sys.argv[1]) if len(sys.argv) > 1 else 15): print(" ", k, v)
