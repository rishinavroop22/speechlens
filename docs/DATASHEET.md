# Datasheet: SpeechLens Contrastive Delivery Dataset

Following the "Datasheets for Datasets" template (Gebru et al., 2021).

## Motivation
- **Purpose.** Evaluate systems that compare a spoken delivery against a well-delivered reading of the *same text* and must locate each delivery flaw in time. No public dataset pairs good and bad deliveries of identical text with time-exact labels.
- **Creators.** The SpeechLens team, for the Multimodal AI Hackathon 2026 (Track C).

## Composition
| Part | Clips | Notes |
|---|---|---|
| Gold readings | 5 | Public-domain presidential speech excerpts (77–111 s each, 120–197 words) |
| Single-flaw clips | 275 | 11 flaw types × 5 severities × 5 speeches |
| Mixed-flaw clips | 40 | 8 levels per speech, 2 mild flaws (level 1) up to 6 severe flaws (level 8) |
| Reading-error mixes | 15 | One skipped-word or restart flaw plus two prosodic flaws, 3 per speech |
| Controls | 20 | Unchanged gold, voice shifted ±4 semitones, 12 dB quieter: must yield no findings |
| Team recordings | 36 | 3 people × 3 passages × 4 takes (1 clean reference take + 3 with scripted flaws) |

- **Flaw types.** Rushed, dragging, monotone, trailing off, missing pauses, awkward pause, filler (uh/um), muffled articulation, repeated onset (stutter), skipped words, restarted phrase.
- **Severity scale.** 1 = near-perfect (inside natural take-to-take variation) to 5 = egregious; each type's parameter per level is listed in `speechlens/inject.py` (`P`, `COUNT`).
- **Labels.** Per clip `data/dataset/labels/<id>.json`: every flaw with type, severity, start/end seconds, word indices, generation parameters, and the word timeline of the new audio. Synthetic labels are exact to the sample, since we performed every edit; human labels come from the recording script, verified by an independent manipulation check (`results/manipulation_check.json`: 58 of 90 scripted flaws were performed).
- **Speakers.** 5 male US presidents (gold) and 3 team members (human takes). **Known gap:** no female gold speaker; the speaker-normalisation is tested through ±4 semitone controls and the team recordings, not through a gender-balanced corpus.
- **Format.** 16 kHz mono; WAV as built, FLAC as distributed. About 9 hours of audio in total.

## Collection process
- Gold audio and transcripts: Miller Center, University of Virginia (US federal government works, public domain). An excerpt of ~75 s per speech is located by forced alignment and cut at sentence boundaries.
- Synthetic flaws: the Flaw Injection Engine (`speechlens/inject.py`) edits the gold audio with Praat PSOLA (timing, pitch), gain ramps, low-pass filtering, speaker-matched filler synthesis, onset repetition, word deletion and phrase repetition. Every random choice is seeded from the clip id (CRC32), and Praat's own random generator is seeded, so a rebuild is bit-identical.
- Team recordings: phone and laptop microphones in ordinary rooms, read from a generated script (`docs/recording_sheet.md`) that specifies which flaw to perform where.

## Preprocessing
Peak normalisation of the gold audio to 0.95; no denoising. Transcripts are normalised for alignment (numbers, dates, dashes) by `speechlens/text.py`.

## Uses
- **Intended.** Benchmarking temporal grounding of delivery flaws, contrastive prosody analysis, robustness studies (see `tools/stress.py`).
- **Not intended.** Judging real people's speaking ability without a human in the loop; training speaker-identification systems.

## Distribution and licence
Audio derived from public-domain recordings; labels, code and team recordings released with the repository (link in the README). Team members consented to the release of their recordings.

## Maintenance
Rebuild with `tools/build_dataset.py` + `tools/extend_dataset.py` (deterministic). Issues and additions via the GitHub repository.
