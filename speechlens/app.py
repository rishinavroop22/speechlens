"""FastAPI server: analysis API + static dashboard.

Run:  uvicorn speechlens.app:app --port 8000
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import audio
from .align import model_available
from .explain import list_rubrics
from .pipeline import evaluate

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "data" / "library"
DATASET = ROOT / "data" / "dataset"
UPLOADS = ROOT / ".cache" / "uploads"
EVAL = ROOT / "results"
WEB = ROOT / "web"

app = FastAPI(title="SpeechLens", version="1.0")


def _library():
    out = []
    for d in sorted(LIB.glob("*/meta.json")):
        m = json.loads(d.read_text())
        m["id"] = d.parent.name
        m["transcript"] = (d.parent / "transcript.txt").read_text().strip()
        out.append(m)
    return out


def _manifest():
    p = DATASET / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {"clips": []}


def peaks(y: np.ndarray, per_sec: int = 100) -> list[float]:
    hop = audio.SR // per_sec
    n = len(y) // hop
    if n == 0:
        return []
    m = np.abs(y[: n * hop]).reshape(n, hop).max(axis=1)
    m = m / (m.max() + 1e-9)
    return [round(float(v), 3) for v in m]


@app.get("/api/status")
def status():
    return {"aligner": "wav2vec2-ctc" if model_available() else "fallback-energy",
            "library": len(_library()), "clips": len(_manifest()["clips"])}


@app.get("/api/library")
def library():
    return _library()


@app.get("/api/rubrics")
def rubrics():
    return list_rubrics()


@app.get("/api/samples")
def samples():
    return _manifest()


@app.get("/api/evaluation")
def evaluation():
    p = EVAL / "evaluation.json"
    if not p.exists():
        return JSONResponse({"available": False})
    return json.loads(p.read_text()) | {"available": True}


@app.get("/media/{kind}/{name:path}")
def media(kind: str, name: str):
    base = {"library": LIB, "dataset": DATASET / "clips", "uploads": UPLOADS}.get(kind)
    if base is None:
        raise HTTPException(404)
    p = (base / name).resolve()
    if base.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/analyze")
async def analyze(
    rubric: str = Form("declamation"),
    reference_id: str | None = Form(None),
    reference_text: str | None = Form(None),
    participant_text: str | None = Form(None),
    sample_id: str | None = Form(None),
    reference_file: UploadFile | None = File(None),
    participant_file: UploadFile | None = File(None),
):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    truth = None
    # ---- reference
    if sample_id:
        clip = next((c for c in _manifest()["clips"] if c["id"] == sample_id), None)
        if clip is None:
            raise HTTPException(404, "Unknown sample.")
        reference_id = clip["source"]
        truth = clip["labels"]
        if clip.get("reference_clip"):
            reference_file = None
            ref_y = audio.load(DATASET / "clips" / f"{clip['reference_clip']}.wav")
            ref_text = clip["transcript"]
            ref_url = f"/media/dataset/{clip['reference_clip']}.wav"
            participant_text = clip["transcript"]
            reference_id = None
    if reference_file is not None and reference_file.filename:
        ref_y = audio.load(await _save(reference_file))
        ref_text = reference_text
        if not ref_text:
            raise HTTPException(400, "Add the transcript for the reference recording.")
        ref_url = None
    elif sample_id and reference_id is None:
        pass  # human take: reference is the same person's clean take, set above
    elif reference_id:
        ref_dir = LIB / reference_id
        if not ref_dir.exists():
            raise HTTPException(404, "Unknown reference speech.")
        ref_y = audio.load(ref_dir / "audio.wav")
        ref_text = (ref_dir / "transcript.txt").read_text()
        ref_url = f"/media/library/{reference_id}/audio.wav"
    else:
        raise HTTPException(400, "Choose a reference speech or upload one.")
    # ---- participant
    if sample_id:
        par_path = DATASET / "clips" / f"{sample_id}.wav"
        par_url = f"/media/dataset/{sample_id}.wav"
    elif participant_file is not None and participant_file.filename:
        par_path = await _save(participant_file)
        par_url = f"/media/uploads/{par_path.name}"
    else:
        raise HTTPException(400, "Upload the participant recording.")
    par_y = audio.load(par_path)
    if par_path.suffix != ".wav" or sample_id is None:
        wav = UPLOADS / (hashlib.sha1(par_y.tobytes()).hexdigest()[:12] + ".wav")
        audio.save(wav, par_y)
        par_url = f"/media/uploads/{wav.name}"
    try:
        rep = evaluate(ref_y, ref_text, par_y, participant_text or None, rubric)
    except ValueError as e:
        raise HTTPException(422, str(e))
    rep["audio_url"] = par_url
    rep["reference_url"] = ref_url
    rep["peaks"] = peaks(par_y)
    rep["truth"] = truth
    return rep


async def _save(f: UploadFile) -> Path:
    data = await f.read()
    name = hashlib.sha1(data).hexdigest()[:12] + Path(f.filename or "x.wav").suffix.lower()
    p = UPLOADS / name
    p.write_bytes(data)
    return p


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
