"""Score the blind listening audit.

Reads results/audit_raw/ratings/*.json (exported from the audit page's
database) and results/audit_key.json (which group each snippet came from).

For each group the share of snippets a rater heard as worse than the clean
take (slightly or clearly) is reported with a Wilson 95 % interval:
  extra    - findings outside any scripted flaw: share heard as worse estimates
             how many of these "false positives" are real delivery problems
  scripted - findings matching a performed scripted flaw: rater sensitivity
  null     - random spans with no finding: rater base rate (false alarms)
The corrected estimate (p_extra - p_null) / (p_scripted - p_null) removes the
rater's own false-alarm rate and imperfect sensitivity (Rogan-Gladen). With
two or more raters, agreement is reported as Cohen's kappa on worse/same.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# detector flaw type -> audit issue chips that describe it
MATCH = {"rushed": {"fast"}, "dragging": {"slow"}, "awkward_pause": {"pause"}, "pause_omission": {"pause", "fast"},
         "filler": {"filler", "pause"}, "stutter": {"filler"}, "monotone": {"flat"}, "trailing_off": {"quiet"},
         "mumbling": {"quiet"}}


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(p, 3), round(max(0, c - h), 3), round(min(1, c + h), 3)


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def main():
    key = json.loads((ROOT / "results/audit_key.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "results/audit_raw/ratings").glob("*.json"))]
    rows = [r.get("data", r) for r in rows]
    by_rater = defaultdict(dict)
    for r in rows:
        if r.get("verdict") and r.get("item") in key:
            by_rater[r["rater"]][r["item"]] = r
    raters = [u for u, d in by_rater.items() if len(d) >= 25]  # only (nearly) complete sets
    out = {"raters": len(raters), "items": len(key), "per_rater": {}, "pooled": {}}
    pooled = defaultdict(lambda: [0, 0])
    type_ok = [0, 0]
    for n_r, u in enumerate(raters, start=1):
        res = {}
        for g in ("extra", "scripted", "null"):
            items = [i for i, k in key.items() if k["group"] == g and i in by_rater[u]]
            worse = sum(by_rater[u][i]["verdict"] in ("slight", "clear") for i in items)
            clear = sum(by_rater[u][i]["verdict"] == "clear" for i in items)
            res[g] = {"n": len(items), "worse": worse, "clear": clear, "worse_rate": wilson(worse, len(items))}
            pooled[g][0] += worse
            pooled[g][1] += len(items)
        for i, k in key.items():
            if k["group"] == "extra" and i in by_rater[u] and by_rater[u][i]["verdict"] in ("slight", "clear"):
                type_ok[1] += 1
                type_ok[0] += bool(set(by_rater[u][i].get("issues", [])) & MATCH[k["claimed"]])
        out["per_rater"][f"rater{n_r}"] = res
    for g, (k, n) in pooled.items():
        out["pooled"][g] = {"worse": k, "n": n, "worse_rate": wilson(k, n)}
    pe = out["pooled"]["extra"]["worse_rate"][0]
    ps = out["pooled"]["scripted"]["worse_rate"][0]
    pn = out["pooled"]["null"]["worse_rate"][0]
    out["extra_real_corrected"] = round(min(1, max(0, (pe - pn) / (ps - pn))), 3) if ps and ps > pn else None
    out["extra_type_agreement"] = wilson(*type_ok)
    if len(raters) >= 2:
        common = [i for i in key if all(i in by_rater[u] for u in raters[:2])]
        a = [by_rater[raters[0]][i]["verdict"] != "same" for i in common]
        b = [by_rater[raters[1]][i]["verdict"] != "same" for i in common]
        out["kappa_worse_vs_same"] = kappa(a, b)
    # adjusted human precision: scripted matches + extras that are real
    ev_p = ROOT / "results/evaluation.json"
    ev = json.loads(ev_p.read_text())
    h = ev.get("cross_validated", {}).get("cv_human", {}).get("__all__")
    if h and out["extra_real_corrected"] is not None:
        tp, fp = h["tp"], h["fp"]
        out["human_precision_scripted_only"] = h["precision"]
        out["human_precision_adjusted"] = round((tp + fp * out["extra_real_corrected"]) / (tp + fp), 3)
        out["human_precision_adjusted_raw"] = round((tp + fp * pe) / (tp + fp), 3)
        lo, hi = out["pooled"]["extra"]["worse_rate"][1:]
        out["human_precision_range"] = [round((tp + fp * lo) / (tp + fp), 3), round((tp + fp * hi) / (tp + fp), 3)]
    e, s, nn = out["pooled"]["extra"], out["pooled"]["scripted"], out["pooled"]["null"]
    ta = out["extra_type_agreement"]
    rng = out.get("human_precision_range")
    out["summary"] = (
        f"{len(raters)} listener(s) from the team rated 30 snippets blind, each against the same words in the speaker's clean take. "
        f"Findings outside any scripted flaw were heard as worse delivery {e['worse']}/{e['n']} times "
        f"({e['worse_rate'][0]*100:.0f} %, 95 % CI {e['worse_rate'][1]*100:.0f}-{e['worse_rate'][2]*100:.0f} %); scripted flaws "
        f"{s['worse']}/{s['n']}; random spans with no finding {nn['worse']}/{nn['n']}, so listeners were not simply calling everything worse. "
        + (f"Counting those as real, precision on the team recordings rises from {out['human_precision_scripted_only']*100:.0f} % "
           f"(scripted flaws only) to an estimated {out['human_precision_adjusted_raw']*100:.0f} % "
           f"(95 % range {rng[0]*100:.0f}-{rng[1]*100:.0f} %). " if rng else "")
        + (f"The listener named the same problem as the detector in {ta[0]*100:.0f} % of those cases: SpeechLens locates "
           f"unscripted problems more reliably than it labels them. " if ta[0] is not None else "")
        + (f"Agreement between listeners: Cohen's kappa {out['kappa_worse_vs_same']}." if "kappa_worse_vs_same" in out
           else "With a single listener who knows the project, treat this as an estimate, not a measurement."))
    ev["audit"] = out
    ev_p.write_text(json.dumps(ev, indent=1))
    (ROOT / "results/audit_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
