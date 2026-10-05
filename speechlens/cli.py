"""Command line: python -m speechlens.cli REF_AUDIO REF_TXT PAR_AUDIO [--par-text T] [--rubric R] [--json OUT]"""
import argparse
import json

from .pipeline import evaluate_files


def main():
    ap = argparse.ArgumentParser(description="Score a delivery against a reference reading of the same text.")
    ap.add_argument("ref_audio")
    ap.add_argument("ref_text")
    ap.add_argument("par_audio")
    ap.add_argument("--par-text", help="participant transcript, if it differs from the reference")
    ap.add_argument("--rubric", default="declamation")
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args()
    rep = evaluate_files(a.ref_audio, a.ref_text, a.par_audio, a.par_text, a.rubric)
    s = rep["score"]
    print(f"\nOverall {s['overall']}/100 ({s['band']}, {s['rubric']} rubric)")
    for d, v in s["dimensions"].items():
        print(f"  {rep['dimension_labels'][d]:15s} {v:4.1f}")
    print(f"\n{len(rep['regions'])} finding(s):")
    for r in rep["regions"]:
        print(f"\n[{r['type']} sev {r['severity']}] {r['start']:.2f}-{r['end']:.2f}s\n  {r['explanation']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"\nreport written to {a.json}")


if __name__ == "__main__":
    main()
