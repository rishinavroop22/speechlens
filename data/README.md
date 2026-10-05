# SpeechLens contrastive dataset

Paired good/flawed recordings of the same transcripts, with time-bounded labels.

**Audio download:** _link added at submission (Google Drive)_. Unzip into this folder so that `data/library/` and `data/dataset/clips/` exist. Labels, alignments and the manifest are in the repository itself.

## Layout

```
library/<speech>/audio.wav         gold excerpt, 16 kHz mono
library/<speech>/transcript.txt    exact text of the excerpt
library/<speech>/alignment.json    word timings (wav2vec2 CTC forced alignment)
library/<speech>/meta.json         speaker, year, source URL, licence, duration, median F0
dataset/clips/<clip>.wav           flawed and control recordings
dataset/labels/<clip>.json         labels + word timings of that recording
dataset/manifest.json              index of every clip
human/script.json                  the recording script for human takes
```

## Clip ids

| Pattern | Meaning |
|---|---|
| `<speech>__gold` | the untouched gold excerpt (control) |
| `<speech>__ctrl_pitch_up4` / `_pitch_down4` / `_quiet12db` | same delivery, voice shifted 4 semitones or 12 dB quieter (controls; no flaw) |
| `<speech>__<flaw>_s<1-5>` | one injected flaw at severity 1 (near-perfect) to 5 (egregious) |
| `<speech>__mix<1-8>` | 2-6 injected flaws; level 1 = two mild, 8 = six severe |
| `human__<name>_<speech>_<A-D>` | team recording; A = clean, B/C/D = scripted flaws |

## Label format

```json
{"type": "rushed", "severity": 3, "start": 12.31, "end": 15.84,
 "word_start": 34, "word_end": 41, "params": {"duration_factor": 0.72}}
```

`start`/`end` are seconds in that clip; `word_start`/`word_end` index the transcript tokens. For synthetic clips the times are exact (computed from the edit itself). For human clips they come from forced alignment of the take and the scripted word positions; severity is fixed at 3.

## Flaw types and severity parameters

| Flaw | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| rushed (duration factor) | 0.88 | 0.80 | 0.72 | 0.64 | 0.55 |
| dragging | 1.12 | 1.24 | 1.36 | 1.48 | 1.62 |
| monotone (pitch excursion removed) | 35 % | 55 % | 70 % | 85 % | 97 % |
| trailing_off (dB at end) | 4 | 7 | 10 | 14 | 18 |
| pause_omission (pause kept) | 60 % | 45 % | 30 % | 15 % | 4 % |
| awkward_pause (s inserted) | 0.35 | 0.60 | 0.90 | 1.30 | 1.80 |
| filler (s each; count 1-4) | 0.28 | 0.36 | 0.45 | 0.55 | 0.65 |
| mumbling (low-pass Hz) | 5000 | 3500 | 2500 | 1800 | 1200 |
| stutter (repetitions) | 1 | 1 | 2 | 2 | 3 |

## Licence

The speeches are US federal government works (public domain); audio and transcripts via the Miller Center, University of Virginia. The derived clips and labels are released under CC BY 4.0.
