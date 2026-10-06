"""Build a Hugging Face Space (Docker) for the live demo into build/space/.

Contents: the app, the five reference speeches, a curated set of dataset
clips with pre-computed reports (so samples open instantly on a small CPU),
the evaluation results, and a Dockerfile that downloads the alignment model
at build time. Uploaded recordings are analysed live.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

from speechlens import audio  # noqa: E402
from speechlens.app import app  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build/space"
PER_SPEECH = ["gold", "mix2", "mix4", "mix6", "mix8", "rushed_s4", "dragging_s4", "monotone_s4", "trailing_off_s4",
              "filler_s4", "mumbling_s4", "awkward_pause_s4", "stutter_s4", "pause_omission_s4", "ctrl_pitch_down4"]

DOCKERFILE = """FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
RUN python tools/get_model.py
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "speechlens.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

README = """---
title: SpeechLens
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
short_description: Finds where a spoken delivery goes wrong, and why.
---

# SpeechLens

Contrastive speech analytics with temporal flaw grounding (Multimodal AI Hackathon 2026, Track C).

Compare a participant's reading with a reference reading of the same text: SpeechLens aligns both word by word,
marks each delivery flaw (pace, pauses, fillers, pitch variety, volume, clarity) with start and end times,
explains it with the measured numbers, and scores it against a rubric.

- **Try a dataset clip** in the left panel (pre-computed, opens instantly), or upload your own recording and
  pick a reference speech. Uploads are analysed live on CPU: allow 20-40 seconds.
- Code, dataset and technical documentation: https://github.com/rishinavroop22/speechlens
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    man = json.loads((ROOT / "data/dataset/manifest.json").read_text())
    by_id = {c["id"]: c for c in man["clips"]}
    keep = [f"{s}__{x}" for s in man["sources"] for x in PER_SPEECH]
    keep += [c["id"] for c in man["clips"] if c["kind"] in ("human", "human_reference")]
    keep = [k for k in keep if k in by_id]

    # code and static files
    shutil.copytree(ROOT / "speechlens", OUT / "speechlens", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "web", OUT / "web")
    (OUT / "tools").mkdir(parents=True)
    shutil.copy(ROOT / "tools/get_model.py", OUT / "tools/get_model.py")
    shutil.copy(ROOT / "requirements.txt", OUT / "requirements.txt")
    (OUT / "Dockerfile").write_text(DOCKERFILE)
    (OUT / "README.md").write_text(README)
    (OUT / ".gitattributes").write_text("*.wav filter=lfs diff=lfs merge=lfs -text\n*.flac filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.woff2 filter=lfs diff=lfs merge=lfs -text\n")
    # data
    shutil.copytree(ROOT / "data/library", OUT / "data/library")
    (OUT / "data/dataset/clips").mkdir(parents=True)
    for k in keep:
        src = audio.clip_path(ROOT / "data/dataset/clips", k)
        dst = OUT / "data/dataset/clips" / f"{k}.flac"
        if src.suffix == ".flac":
            shutil.copy(src, dst)
        else:
            import soundfile as sf
            y, sr = sf.read(src)
            sf.write(dst, y, sr, format="FLAC", subtype="PCM_16")
    man2 = dict(man, clips=[by_id[k] for k in keep])
    (OUT / "data/dataset/manifest.json").write_text(json.dumps(man2))
    (OUT / "results").mkdir()
    shutil.copy(ROOT / "results/evaluation.json", OUT / "results/evaluation.json")

    # pre-computed reports, produced by the real endpoint
    (OUT / "data/precomputed").mkdir(parents=True)
    client = TestClient(app)
    n = 0
    reports = {}
    for k in keep:
        if by_id[k]["kind"] == "human_reference":
            continue
        r = client.post("/api/analyze", data={"sample_id": k, "rubric": "declamation"})
        r.raise_for_status()
        rep = r.json()
        rep.pop("score", None)
        rep["audio_url"] = f"/media/dataset/{k}.flac"
        if rep.get("reference_url", "").startswith("/media/dataset/"):
            rep["reference_url"] = f"/media/dataset/{rep['reference_url'].rsplit('/', 1)[1].rsplit('.', 1)[0]}.flac"
        reports[k] = rep
        n += 1
    # two bundle files, each under 20 MB, instead of one file per clip
    ids = sorted(reports)
    for i, part in enumerate((ids[: len(ids) // 2], ids[len(ids) // 2:]), start=1):
        (OUT / "data/precomputed" / f"reports_{i}.json").write_text(
            json.dumps({k: reports[k] for k in part}, separators=(",", ":")))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{len(keep)} clips, {n} pre-computed reports, {sum(1 for f in OUT.rglob('*') if f.is_file())} files, "
          f"{size / 1e6:.0f} MB -> {OUT}")


if __name__ == "__main__":
    main()
