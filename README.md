# SpeechLens

**Contrastive speech analytics with temporal flaw grounding.**
Multimodal AI Hackathon 2026, Track C.

SpeechLens compares a participant's spoken delivery against a well-delivered reference reading of the *same transcript*. It aligns both readings word by word, measures how pace, pausing, pitch, loudness, articulation and fluency differ, marks every delivery flaw with exact start and end times, and explains each one with the numbers behind it and a concrete fix. A rubric turns the findings into a reproducible score.

It is trained and tested on a **contrastive dataset we built**: five public-domain presidential speeches, each mirrored by a graded spectrum of flawed recordings of the same words, from near-perfect to egregious, with sample-exact labels.

| | |
|---|---|
| Dashboard | upload audio + transcript, see flaw regions on a time-aligned overlay, A/B-listen participant vs reference |
| Dataset | 5 gold speeches, 285 synthetic spectrum clips, 20 controls, scripted human recordings |
| Flaw types | rushed, dragging, monotone, trailing off, missing pauses, awkward pause, filler, muffled articulation, repeated onset |
| Runs on | CPU only, no GPU, no cloud API, no language model; identical output on every run |

---

## Quick start

```bash
git clone <this repo> && cd speechlens
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/get_model.py                               # downloads the 378 MB alignment model
uvicorn speechlens.app:app --port 8000                  # open http://localhost:8000
```

Prerequisites: Python 3.10+, `ffmpeg` on PATH (for MP3/M4A input). With Docker instead:

```bash
docker build -t speechlens . && docker run -p 8000:8000 speechlens
```

The dataset audio is distributed separately (see [Dataset](#dataset)); unpack it into `data/` to use the sample clips and re-run the evaluation.

### Command line

```bash
# score one recording against a reference
python -m speechlens.cli reference.wav reference.txt participant.m4a --rubric declamation

# rebuild the dataset from raw speeches in data/raw/
python tools/build_dataset.py --excerpt-sec 75

# reproduce every number in the evaluation section
python tools/evaluate.py --workers 2
```

---

## How it works

```
reference audio + transcript ─┐                          ┌─> regions (type, start, end, severity, z, metrics)
                              ├─ forced alignment ─ features ─ word-paired contrast ─┼─> causal explanations
participant audio (+ text) ───┘   (wav2vec2 CTC)   (10 ms grid)                       └─> rubric score 0-100
```

1. **Forced alignment.** `wav2vec2-base-960h` (ONNX, CPU) produces per-frame character probabilities; an exact CTC Viterbi pass finds the single path that spells the transcript, giving every word a start and end. Boundaries are refined against the energy envelope so word tails and pauses are measured from real silence.
2. **Feature extraction** on a shared 10 ms grid: F0 (two-pass, speaker-adaptive range), intensity, harmonics-to-noise ratio, spectral energy above 2 kHz and 4 kHz, spectral flux, MFCCs. Aggregated per word: articulation rate, pause after, voiced sound inside pauses, pitch range, loudness, articulation energy, recognizer confidence.
3. **Speaker normalization.** Pitch is in semitones relative to each speaker's own median F0 and loudness in dB relative to each speaker's median speech level, so a deep and a high voice reading identically produce the same contours, and recording gain is irrelevant.
4. **Word-paired contrast.** Words of the two readings are paired by transcript position. Each contrast signal is split into a *global* offset (the speaker's overall habit, reported separately) and a *local* deviation, scored as a robust z-score against the reference speaker's own word-to-word variability, smoothed, thresholded and merged into regions.
5. **Causal explanation.** Each region becomes a sentence filled from measured values: what changed, by how much, how unusual that is, why it hurts delivery, and how to fix it. No language model is involved.
6. **Rubric scoring.** Six dimensions (pace, pausing, fluency, pitch variety, volume control, clarity) start at 10 and lose points per region by severity and coverage. Presets weight them for Declamation, Interpretive Reading, Extemporaneous and Persuasive Oratory.

Full detail, formulas and thresholds: [`docs/methodology.md`](docs/methodology.md).

---

## Dataset

### Gold baseline
Opening excerpts (73-110 s) of five speeches, all US federal government works in the public domain, audio and transcripts from the Miller Center, University of Virginia:

| id | Speech | Year |
|---|---|---|
| `fdr` | Franklin D. Roosevelt, address to Congress ("Day of Infamy") | 1941 |
| `jfk` | John F. Kennedy, Inaugural Address | 1961 |
| `lbj` | Lyndon B. Johnson, voting rights ("We Shall Overcome") | 1965 |
| `reagan` | Ronald Reagan, Challenger address | 1986 |
| `obama` | Barack Obama, Inaugural Address | 2009 |

### The bad spectrum
**Synthetic (exact labels).** The Flaw Injection Engine (`speechlens/inject.py`) edits the gold recording with Praat PSOLA resynthesis, which changes timing and pitch while keeping the speaker's own voice. Because we make every edit ourselves, each label is exact to the sample. Per speech:
- 45 single-flaw clips: 9 flaw types x 5 severities (1 = near-perfect, 5 = egregious);
- 8 mixed clips forming a gradient from two mild flaws to six severe ones;
- 4 controls with no flaw: the untouched gold, the voice shifted +4 and -4 semitones, and 12 dB quieter.

**Human (scripted labels).** Team members read three of the excerpts four times each: one clean take and three takes with flaws performed at positions fixed by a recording script ([`docs/recording_sheet.md`](docs/recording_sheet.md)). Labels come from aligning each take and mapping the scripted words to times.

### Files
```
data/library/<id>/      audio.wav, transcript.txt, alignment.json, meta.json
data/dataset/clips/     <clip>.wav
data/dataset/labels/    <clip>.json   labels + word timings of the flawed recording
data/dataset/manifest.json
```
Audio download: see the link in [`data/README.md`](data/README.md).

---

## Evaluation

All numbers come from running the full pipeline on raw audio, including forced alignment of the flawed recording (no oracle timings). Reproduce with `python tools/evaluate.py`; the dashboard's Evaluation tab renders `results/evaluation.json`.

Headline (285 clips, 5 speeches, all on raw audio):

| | |
|---|---|
| Flaw localization F1 (tIoU >= 0.3) | **0.73** (precision 0.75, recall 0.71) |
| Median boundary error of matched flaws | **150 ms** |
| False findings on unflawed controls (incl. voice shifted ±4 semitones, -12 dB) | **2 in 29 min** |
| Score vs mixed-flaw level, Spearman ρ | **-0.954** |
| Severity error, leave-one-speech-out | **0.493 levels**, 91% within one level |
| Re-run on a fresh process | byte-identical |

Per flaw type (tIoU >= 0.3):

| Flaw | Precision | Recall | F1 | Mean tIoU | Boundary err (ms) |
|---|---|---|---|---|---|
| Rushed | 0.95 | 0.86 | 0.91 | 0.80 | 213 |
| Dragging | 0.78 | 0.69 | 0.73 | 0.82 | 248 |
| Monotone | 1.00 | 0.85 | 0.92 | 0.73 | 495 |
| Trailing off | 0.88 | 0.68 | 0.77 | 0.67 | 475 |
| Missing pauses | 0.66 | 0.62 | 0.64 | 0.99 | 3 |
| Awkward pause | 0.78 | 0.77 | 0.77 | 0.80 | 117 |
| Filler | 0.70 | 0.72 | 0.71 | 0.56 | 151 |
| Muffled articulation | 1.00 | 0.52 | 0.69 | 0.77 | 370 |
| Repeated onset | 0.51 | 0.61 | 0.55 | 0.84 | 21 |
| **All flaws** | 0.75 | 0.71 | 0.73 | 0.74 | 150 |

Detection rises with severity as designed: severity 1 is near-perfect by construction and mostly passes; from severity 3 up, pace, monotone, pause and fluency flaws are found 60-100 % of the time (full table in the dashboard's Evaluation tab).

---

## Limitations

- The reference must read the same text. Without a reference, SpeechLens falls back to corpus norms, which are coarser.
- Synthetic flaws are cleaner than human ones; human-recording results are reported separately for that reason.
- Muffled articulation is measured through high-frequency energy, so a low-pass microphone or heavy compression can mimic it. The global/local split absorbs a constant channel difference but not one that changes mid-recording.
- Fillers and repeated onsets are found inside pauses; a filler fused into a word with no gap can be missed.
- The alignment model is English-only.
- Thresholds were set on the synthetic data, and human recordings are the out-of-distribution check.

## Attributions

- Speech audio and transcripts: Miller Center, University of Virginia; the speeches are US federal government works (public domain).
- Alignment model: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) (Apache-2.0), ONNX export by [Xenova](https://huggingface.co/Xenova/wav2vec2-base-960h).
- Praat via [Parselmouth](https://github.com/YannickJadoul/Parselmouth) (GPL-3.0); librosa (ISC); onnxruntime (MIT); FastAPI (MIT).
- Font: Archivo (SIL Open Font License).
