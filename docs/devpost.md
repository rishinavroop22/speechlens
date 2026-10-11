# SpeechLens — Devpost submission text

**Tagline:** Hear exactly where a delivery goes wrong, why, and what it sounds like fixed.

**Track:** C — Contrastive Speech Analytics & Temporal Flaw Grounding

**Links:** Live demo https://rishinavroop22.github.io/speechlens/ · Code https://github.com/rishinavroop22/speechlens · Video: (YouTube link) · Dataset: (Google Drive link)

---

## Inspiration
Every elocution or debate judge has said "work on your pacing" to a speaker, and every speaker has wondered *where*. Delivery feedback is subjective, late and vague, while the thing being judged, a voice reading a known text, is measurable. We wanted a system that listens like a coach who holds the reference recording in their head: it compares the two readings word by word and points at the exact second the delivery departs from it.

## What it does
SpeechLens compares a participant's reading with a well-delivered reference reading of **the same text**.
- It aligns both readings word by word and measures pace, pauses, pitch, loudness, articulation and fluency on the same scale for any voice.
- It marks **11 kinds of delivery flaw** with exact start and end times: rushed, dragging, monotone, trailing off, missing pauses, awkward pauses, fillers, muffled articulation, repeated onsets, and two reading errors, **skipped words** and **restarted phrases**.
- Every finding comes with the numbers behind it ("articulation 7.1 syll/s against 5.1 in the reference, +40 %, 1.6× the reference speaker's natural variation"), why it hurts the delivery, and a fix. No language model writes these, so the same audio always gives the same result.
- **Hear it fixed:** for any finding, SpeechLens plays the participant's own voice with only the named cause corrected. That is also how we *prove* the explanations are causal (see below).
- **Practice this part:** record a retake of one passage and get before/after numbers against the same reference.
- A coach's summary with priorities and drills, a voice-quality profile (jitter, shimmer, HNR, CPPS), a progress chart across takes, a printable report, four competition rubrics (Declamation, Interpretive Reading, Extemporaneous, Persuasive Oratory), microphone recording, and a warning when the recording is too noisy to trust.

## How we built it
- **Dataset.** No public dataset pairs good and bad deliveries of identical text with time-exact labels, so we built one. Five public-domain presidential speeches are the gold standard. A **Flaw Injection Engine** edits them with Praat PSOLA in the speaker's own voice. 350 synthetic clips cover 11 flaw types × 5 severities, mixed-flaw gradients and unflawed controls (voice shifted ±4 semitones, 12 dB quieter), and every label is exact to the sample. Three of us also recorded 36 takes with scripted flaws.
- **Pipeline.** wav2vec2 CTC forced alignment (ONNX, exact Viterbi, CPU only) → speaker-normalised features on a 10 ms grid (Praat F0, intensity, HNR, STFT band energies, MFCC) → word-paired contrast that separates the speaker's overall habit from local deviations, scored against the reference speaker's own variability → data-fitted significance gates → template explanations → rubric score.
- **Reading errors.** A skipped word leaves the aligner squeezing letters into a few frames with near-zero confidence. A restart adds speech at one boundary. A short recognition pass over just that neighbourhood confirms each one and becomes the evidence shown to the user ("recognised here: *to our secretary to our secretary of state*").
- **Stack.** Python, FastAPI, numpy/scipy, Parselmouth (Praat), librosa, onnxruntime, and a no-build vanilla-JS dashboard with canvas timelines. Dashboard on GitHub Pages; live uploads analysed by the same pipeline in a Docker container on Azure Container Apps (image built by GitHub Actions). Runs locally with one command or Docker.

## How we know it works
Every number runs the full pipeline on raw audio, and thresholds are cross-validated (leave-one-speech-out / leave-one-speaker-out).
- Locating synthetic flaws (11 types): **F1 0.66** (95 % CI 0.62–0.70), precision 0.82; median boundary error **150 ms**.
- Unflawed speech, even with the voice shifted four semitones: **0.03 false findings per minute**.
- Score falls steadily with flaw level: Spearman **ρ = −0.95**; severity within one level 93 % of the time.
- **Why the reference matters:** the same detector without a reference reaches F1 0.07, and only 0.14 even with thresholds tuned on the test answers.
- **Counterfactual test:** repairing only the named cause makes the finding disappear in **91 %** of cases, so the explanations name real causes.
- **New voices:** on 20 unseen LibriSpeech readers, 10 women and 10 men, with every threshold frozen: **F1 0.85** (women 0.86, men 0.85).
- **Feature evidence:** each flaw is carried by exactly the signal its explanation names (e.g. muffled speech by high-frequency energy, Cohen's d = −5.1).
- **Stress test:** robust to MP3 compression, telephone audio, ±10 % tempo and moderate echo. Background noise is the honest weakness, so the app measures it, compensates partly and warns.
- **Real people:** on our own recordings it finds 53 % of verified performed flaws. A manipulation check and a blind listening audit show most "extra" findings are real slips (estimated precision ~0.80).
- Byte-identical results on every rerun; 25 unit tests and CI.

## Challenges we ran into
- Natural take-to-take variation: a person never reads the same sentence twice at the same pace. We fitted per-type significance gates on data and cross-validated them, which doubled precision on human recordings.
- Phone recordings: automatic gain control erased our scripted fades, and "mumbles" were whispered instead. We added a manipulation check so we only score flaws that were really performed.
- Making the injection deterministic: Praat's overlap-add draws random numbers, so we seed its generator per call to make rebuilds bit-identical.
- Background noise breaks contrastive comparison. We found it with our own stress test, tried channel matching, and chose to warn honestly where the fix is partial.

## Accomplishments that we're proud of
- An explanation you can *listen to*: "Hear it fixed" turns every claim into a testable intervention, and the test passes in 91 % of cases.
- A labelled contrastive dataset with exact ground truth, a datasheet, and a one-command rebuild.
- A system that runs on a laptop CPU with no cloud API and no language model, yet catches reading errors with readable evidence.

## What we learned
Contrast is the signal: the same measurements without a reference are close to useless. And the evaluation design (controls, ablations, interventions, stress, listening audits) taught us more about the system than any single accuracy number.

## What's next
Accent-diverse and spontaneous-speech evaluation, noise-robust features, more languages through multilingual alignment models, and a mobile practice app for speech coaches.

## Built with
python · fastapi · praat · parselmouth · wav2vec2 · onnxruntime · librosa · numpy · scipy · javascript · html-canvas · github-pages · docker · azure

## Team
(Full names of all members, each with a Devpost account)
