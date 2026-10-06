"""Figures for the README and the technical document, from results/evaluation.json.

  docs/figures/recall_by_severity.{svg,png}  detection rate per flaw type and severity
  docs/figures/score_by_level.{svg,png}      mean score across the mixed-flaw gradient

Usage: python tools/figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs/figures"
NAMES = {"rushed": "Rushed", "dragging": "Dragging", "monotone": "Monotone", "trailing_off": "Trailing off",
         "pause_omission": "Missing pauses", "awkward_pause": "Awkward pause", "filler": "Filler",
         "mumbling": "Muffled", "stutter": "Repeated onset", "skipped_word": "Skipped words", "restart": "Restart"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none"})


def main():
    e = json.loads((ROOT / "results/evaluation.json").read_text())
    rs = e["recall_by_severity"]
    types = [t for t in NAMES if t in rs]
    M = np.array([[np.nan if v is None else v for v in rs[t]] for t in types], float)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.imshow(np.nan_to_num(M), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.6 else "#1f2430")
    ax.set_xticks(range(5), [str(k) for k in range(1, 6)])
    ax.set_yticks(range(len(types)), [NAMES[t] for t in types])
    ax.set_xlabel("Injected severity (detection rate, %)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"recall_by_severity.{ext}", dpi=200)
    plt.close(fig)

    m = e["score_validity"]["mean_score_by_mix_level"]
    ks = sorted(m, key=int)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.bar(ks, [m[k] for k in ks], color="#2f6fdf", width=0.65)
    for k in ks:
        ax.text(k, m[k] + 1.5, f"{m[k]:.0f}", ha="center", fontsize=7)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Mixed-flaw level (1 = two mild, 8 = six severe)")
    ax.set_ylabel("Mean score / 100")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"score_by_level.{ext}", dpi=200)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
