# Model card: SpeechLens

## What it is
A deterministic analysis pipeline, not a trained network: forced alignment with a pretrained acoustic model (facebook/wav2vec2-base-960h, ONNX export, run on CPU), Praat/STFT feature extraction, and a rule-based contrast engine whose thresholds, significance gates and severity tables are fitted from the dataset. Explanations are templates filled with measured values; no language model is involved.

## Inputs and outputs
- **In:** a participant recording (any format ffmpeg reads, or the microphone), a reference recording of the same text (or one of five built-in references), the transcript (optional for an uploaded reference: recognised automatically), a rubric.
- **Out:** time-stamped findings of 11 flaw types with severity 1–5, confidence, measurements, the rule that fired and an explanation with a fix; a 0–100 rubric score over six dimensions; a counterfactual repair of each finding; a recording-quality check.

## Performance (details in the README and `results/`)
- Synthetic flaws, leave-one-speech-out: see `results/evaluation.json` (`cross_validated`).
- Team recordings, leave-one-speaker-out: recall about 0.55; precision 0.22 against scripted flaws only, about 0.80 after a blind listening audit.
- Unflawed controls: about 0.03 false findings per minute.
- 20 unseen LibriSpeech speakers (10 women, 10 men), thresholds frozen: F1 0.85 (women 0.86, men 0.85).
- Robustness: stable under MP3 compression, telephone band, ±10 % tempo and moderate reverberation; **degrades sharply with background noise** in the participant recording (`results/stress.json`).

## Limitations and risks
- Needs a reference reading of the same text; English only (alignment model).
- Background noise the reference lacks produces false timing and clarity findings; the app warns above 6 dB of excess noise.
- Gold speakers are all male; on 20 unseen LibriSpeech readers (10 women) with frozen thresholds F1 is 0.85 for women and men alike, but accents and non-read speech are untested.
- Labels of unscripted findings agree with human listeners less often than their locations do.
- Delivery norms are those of the reference: a stylistic choice that differs from the reference is reported as a deviation. The tool supports a coach or judge; it should not be the sole basis of a high-stakes decision.

## Reproducibility
`reproduce.sh` (or `make reproduce`) rebuilds the dataset and every number; output is byte-identical across runs (checked inside the evaluation).
