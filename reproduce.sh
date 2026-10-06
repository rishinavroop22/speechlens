#!/usr/bin/env sh
# Rebuild the dataset and every reported number from the raw speeches.
# Requires data/raw/*.mp3|txt (gold speeches), data/raw/human/* (team takes)
# and the alignment model (python tools/get_model.py). ~6 h on 2 CPU cores
# (the stress test is most of it; skip it with SKIP_STRESS=1).
set -e
python tools/build_dataset.py --excerpt-sec 75        # gold excerpts + synthetic spectrum
python tools/recording_sheet.py                        # human recording script
python tools/label_human.py --merge                    # align + label team takes
python tools/verify_human.py                           # manipulation check
SPEECHLENS_NO_GATE=1 python tools/evaluate.py          # raw findings for threshold fitting
python tools/fit_thresholds.py                         # cross-validated significance gates
python tools/evaluate.py                               # gated findings
python tools/calibrate.py                              # severity tables, leave-one-speech-out
python tools/evaluate.py                               # final numbers -> results/evaluation.json
[ -d results/audit_raw ] && python tools/audit_report.py || true
python tools/ablate.py                                 # ablations + reference-free baseline
python tools/counterfactual_eval.py                    # interventional check of explanations
[ -n "$SKIP_STRESS" ] || python tools/stress.py         # degraded-recording stress test
python -m pytest -q tests
