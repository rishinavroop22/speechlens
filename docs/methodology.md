# Methodology

This document specifies every step of SpeechLens precisely enough to reimplement it. Code references point to `speechlens/`.

## 1. Signal representation

All audio is decoded to 16 kHz mono float32 (`audio.py`). Every feature lives on one 10 ms grid, frame `k` covering `t = 0.01 k` s.

## 2. Forced alignment (`align.py`)

**Acoustic model.** `wav2vec2-base-960h` fine-tuned for English CTC on LibriSpeech, run with onnxruntime on CPU. Input is the waveform normalized to zero mean, unit variance; output is a log-probability matrix `L[t, c]` over 32 symbols (A-Z, apostrophe, word separator `|`, blank) at a 20 ms stride. Audio longer than 20 s is processed in 20 s windows with 1 s of context on each side, and only the centre frames are kept, so memory is flat and results do not depend on audio length.

**Transcript normalization** (`text.py`). Display tokens keep their original spelling; each also gets a model spelling: numbers to words (years as years, a day number after a month name as an ordinal), dashes split words, everything else reduced to A-Z and apostrophe. Punctuation after each token is kept as a pause class: comma, clause (`; : —`), sentence (`. ! ?`).

**Viterbi.** With target sequence `y_1..y_N` (characters with `|` between words), the CTC state sequence is `blank, y_1, blank, y_2, ..., y_N, blank` (`S = 2N+1`). The dynamic program

```
alpha_t(s) = L[t, ext_s] + max( alpha_{t-1}(s), alpha_{t-1}(s-1), alpha_{t-1}(s-2) if ext_s != blank and ext_s != ext_{s-2} )
```

is evaluated exactly (no beam), and the best path is backtracked from the last or second-to-last state. Each word's span is the first to last frame assigned to its characters. Word confidence is the mean posterior `exp(L[t, c])` of the intended characters over those frames.

**Boundary refinement.** CTC fires sharp spikes, accurate at onsets but cutting word tails. Each boundary is extended along the 10 ms RMS envelope while energy stays above -32 dB re. the recording's 95th percentile, by at most 120 ms backwards and 350 ms forwards, never crossing a neighbour.

**Excerpt location** (`tools/build_dataset.py`). For long recordings, a greedy CTC decode of the first 10 minutes is matched against the transcript (`difflib`). The speech is taken to start at the first run of at least five consecutive matching words (shorter runs are introductions and applause), backed up to the start of that sentence. It ends at the sentence end closest to the target length, but not shorter than 70 % of it.

## 3. Features (`features.py`)

| Feature | Definition |
|---|---|
| F0 | Praat autocorrelation pitch, two passes: first 60-700 Hz, then floor = 0.75 x Q25 and ceiling = 1.5 x Q75 of the first pass (De Looze & Hirst). |
| Pitch (st) | `12 log2(F0 / median F0)`; speaker-relative semitones. |
| Intensity (dB) | Praat intensity minus the median over speech frames (frames within 30 dB of the 95th percentile); floored at -40. |
| HNR | Praat cross-correlation harmonicity. |
| Articulation energy | Share of STFT power (512-point, 10 ms hop) above 2 kHz, and above 4 kHz. Undefined in silence. |
| Spectral flux | RMS positive log-spectral difference between frames. |
| MFCC | 13 coefficients from a 40-band mel spectrogram. |

Per word: duration, syllable count (vowel-group heuristic on the normalized spelling), articulation rate = syllables / duration, pause after = gap to next word, *filled pause* = voiced frames with intensity > -12 dB inside a gap longer than 150 ms (counted if > 120 ms), pitch mean and p10-p90 range, mean and peak intensity, HNR, both articulation energies, and recognizer confidence.

## 4. Contrastive analysis (`contrast.py`)

**Word pairing.** The two readings' normalized word sequences are matched with `difflib.SequenceMatcher`; only matched words are compared, so a skipped or inserted word affects only itself.

**Global and local deviation.** For a continuous contrast signal `s_j` over pairs `j`:

```
g   = median_j s_j                           (the speaker's overall habit)
l_j = smooth_3(s_j - g)                      (where delivery departs from the reference)
z_j = l_j / max(floor, 1.4826 * MAD(reference's own signal around its median))
```

The denominator is the reference speaker's natural word-to-word variability, so `z` reads as "this deviation is z times larger than how much the reference itself varies". Local regions are runs of words with `l_j` past a threshold (holes of one word closed, minimum length shown below) whose mean `|z| >= 1.5`. A global offset past its own threshold is reported as one whole-reading finding.

| Flaw | Signal `s_j` | Local threshold | Min words |
|---|---|---|---|
| Rushed / dragging | `log2(rate_participant / rate_reference)`, rates over a 3-word window excluding pauses | `> +0.20` / `< -0.20` (15 %) | 3 |
| Monotone | `log2(range_participant / range_reference)`, pitch p10-p90 range over a 5-word window | `< -0.55` (68 %) | 4 |
| Trailing off | `intensity_participant - intensity_reference` (each speaker-relative) | `< -3.5 dB` | 3 |
| Muffled articulation | mean of `log2` ratios of energy above 2 kHz and above 4 kHz | `< -0.75` | 3 |

**Per-gap flaws.** For each pair of consecutive matched words, with reference pause `p_r` and participant pause `p_p`:

- *Repeated onset*: energy bursts (> -16 dB, >= 30 ms) shorter than 220 ms in the gap before a word that the reference does not have. The first burst's mean MFCC (c1-c12) is compared with the next word's first 120 ms; cosine > 0.85, or two or more bursts, marks a repetition. Also flagged when a word gains two or more internal bursts and lasts > 1.4x the reference.
- *Filler*: voiced sound in the pause exceeding the reference's by >= 150 ms.
- *Awkward pause*: `p_p - p_r >= 0.30 s` and `p_p > 1.8 p_r + 0.15 s`.
- *Missing pause*: a reference pause >= 180 ms kept at < 40 %. Missing pauses within six words of each other merge into one region.

**Conflict resolution.** A mild (severity <= 2) loudness drop overlapping a muffled-articulation region by more than half is dropped, since low-pass filtering itself lowers measured loudness.

**Severity.** Each region's magnitude is mapped to 1-5 through the midpoints of the same parameter table the injection engine uses (for example, a local duration factor of 0.68 sits between the severity-3 value 0.72 and the severity-4 value 0.64, closer to 0.64, so severity 4).

## 5. Explanation (`explain.py`)

A fixed template per flaw type is filled with: location (time and words), the primary metric for reference vs participant with the relative change, the z-score, a one-sentence acoustic or perceptual rationale, and an actionable fix. The same input always yields the same text.

## 6. Scoring

Six dimensions start at 10. Each region subtracts

- span flaws: `severity x (0.55 + 4 x share of words covered)`;
- point flaws (awkward pause, filler, repeated onset): `0.45 x severity` each;
- whole-reading findings: `1.6 x severity`.

The overall score is `10 x` the rubric-weighted mean of the dimensions. Rubric presets (`speechlens/rubrics/*.json`) set the weights and score bands.

## 7. Flaw injection (`inject.py`)

The gold recording is cut at the midpoints of inter-word gaps, and the selected span is edited:

| Flaw | Edit | Severity 1 to 5 |
|---|---|---|
| Rushed | PSOLA duration tier: words x f, pauses x f^1.6 | f = 0.88, 0.80, 0.72, 0.64, 0.55 |
| Dragging | same, f > 1 | 1.12 to 1.62 |
| Monotone | PSOLA pitch tier: excursion around the span median scaled by (1 - a) | a = 0.35 to 0.97 |
| Trailing off | gain ramp over 60 % of the span, then held | -4 to -18 dB |
| Missing pauses | punctuation pauses cut to a share, keeping their centre | 60 % to 4 % kept |
| Awkward pause | silence with the recording's own noise floor, mid-phrase | 0.35 to 1.8 s |
| Filler | the speaker's steadiest 90 ms vowel, PSOLA-stretched with falling pitch; half get a nasal "m" tail | 0.28 to 0.65 s, 1 to 4 per clip |
| Muffled articulation | 6th-order Butterworth low-pass, small gain loss | 5000 to 1200 Hz |
| Repeated onset | first 150 ms of the word repeated before it | 1 to 3 repetitions |

New word boundaries are computed by integrating the same piecewise-linear duration tier that Praat applies (rescaled to the exact resynthesized length), and inserted material shifts later times exactly. All random choices use seeds derived from CRC32 of the clip id, so builds are bit-identical across machines.

## 8. Evaluation protocol (`tools/evaluate.py`)

- **Matching**: predicted and true regions of the same type are paired greedily by temporal IoU, one to one. Precision, recall and F1 are reported at tIoU >= 0.3 and >= 0.5, with mean tIoU and median boundary error of matched pairs. Whole-reading findings are excluded from grounding metrics.
- **Sensitivity**: recall per flaw type per severity on single-flaw clips.
- **Severity error**: predicted minus true severity on matched pairs.
- **Controls**: false findings per minute on the unflawed gold, ±4-semitone and -12 dB versions.
- **Score validity**: Spearman correlation between total injected severity and overall score, and between mixed-gradient level and score.
- **Reproducibility**: a subset is re-run in a fresh process and compared byte for byte.
