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
PRECOMP = ROOT / "data" / "precomputed"
EVAL = ROOT / "results"
WEB = ROOT / "web"

app = FastAPI(title="SpeechLens", version="1.0")
# the hosted static page (GitHub Pages) may call this server for live analysis
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
# analysis inputs by report id, so a finding can be repaired afterwards
CTX: dict[str, tuple] = {}


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
    rep = json.loads(p.read_text()) | {"available": True}
    # later experiments, each written by its own tool
    for key, name in (("stress", "stress.json"), ("ablation", "ablation.json"), ("counterfactual", "counterfactual.json"),
                      ("stats", "stats.json")):
        f = EVAL / name
        if f.exists():
            d = json.loads(f.read_text())
            rep[key] = d.get("summary", d) if key == "counterfactual" else d
    return rep


@app.get("/media/{kind}/{name:path}")
def media(kind: str, name: str):
    base = {"library": LIB, "dataset": DATASET / "clips", "uploads": UPLOADS}.get(kind)
    if base is None:
        raise HTTPException(404)
    p = (base / name).resolve()
    if base.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/flac" if p.suffix == ".flac" else "audio/wav")


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
    auto_text = None
    if sample_id and sample_id in _precomp():
        return _precomputed(sample_id, rubric) | {"report_id": f"s-{sample_id}"}
    # ---- reference
    if sample_id:
        clip = next((c for c in _manifest()["clips"] if c["id"] == sample_id), None)
        if clip is None:
            raise HTTPException(404, "Unknown sample.")
        reference_id = clip["source"]
        truth = clip["labels"]
        if clip.get("reference_clip"):
            reference_file = None
            ref_y = audio.load(audio.clip_path(DATASET / "clips", clip['reference_clip']))
            ref_text = clip["transcript"]
            ref_url = f"/media/dataset/{audio.clip_path(DATASET / 'clips', clip['reference_clip']).name}"
            participant_text = clip["transcript"]
            reference_id = None
    if reference_file is not None and reference_file.filename:
        ref_y = audio.load(await _save(reference_file))
        ref_text = reference_text
        auto_text = None
        if not ref_text:
            if not model_available():
                raise HTTPException(400, "Add the transcript for the reference recording.")
            from .align import transcribe
            ref_text = auto_text = transcribe(ref_y)  # no transcript given: recognise it
            if len(ref_text.split()) < 5:
                raise HTTPException(422, "Could not recognise enough words in the reference recording.")
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
        par_path = audio.clip_path(DATASET / "clips", sample_id)
        par_url = f"/media/dataset/{par_path.name}"
    elif participant_file is not None and participant_file.filename:
        par_path = await _save(participant_file)
        par_url = f"/media/uploads/{par_path.name}"
    else:
        raise HTTPException(400, "Upload the participant recording.")
    par_y = audio.load(par_path)
    if par_path.suffix not in (".wav", ".flac") or sample_id is None:
        wav = UPLOADS / (hashlib.sha1(par_y.tobytes()).hexdigest()[:12] + ".wav")
        audio.save(wav, par_y)
        par_url = f"/media/uploads/{wav.name}"
    try:
        rep = evaluate(ref_y, ref_text, par_y, participant_text or None, rubric, voice=True)
    except ValueError as e:
        raise HTTPException(422, str(e))
    rid = hashlib.sha1(par_y.tobytes() + ref_y.tobytes() + (participant_text or "").encode()).hexdigest()[:16]
    CTX[rid] = (ref_y, ref_text, par_y, participant_text or None)
    while len(CTX) > 32:
        CTX.pop(next(iter(CTX)))
    rep["report_id"] = rid
    rep["audio_url"] = par_url
    rep["reference_url"] = ref_url
    rep["peaks"] = peaks(par_y)
    rep["truth"] = truth
    rep["auto_transcript"] = auto_text
    return rep


def _sample_ctx(sample_id: str):
    clip = next((c for c in _manifest()["clips"] if c["id"] == sample_id), None)
    if clip is None:
        raise HTTPException(404, "Unknown sample.")
    if clip.get("reference_clip"):
        ref_y, ref_t = audio.load(audio.clip_path(DATASET / "clips", clip["reference_clip"])), clip["transcript"]
    else:
        ref_y, ref_t = audio.load(LIB / clip["source"] / "audio.wav"), (LIB / clip["source"] / "transcript.txt").read_text()
    return ref_y, ref_t, audio.load(audio.clip_path(DATASET / "clips", sample_id)), clip.get("transcript")


@app.get("/api/fix/{report_id}/{index}")
def fix(report_id: str, index: int):
    """Counterfactual repair of one finding: the participant's audio with only
    the stated cause moved to the reference value (speechlens/counterfactual.py)."""
    from .contrast import compare
    from .counterfactual import repair, snippet
    from .pipeline import analyze_cached
    ctx = _sample_ctx(report_id[2:]) if report_id.startswith("s-") else CTX.get(report_id)
    if ctx is None:
        raise HTTPException(404, "Analysis expired - analyse the recording again.")
    ref_y, ref_t, par_y, par_t = ctx
    from .pipeline import prepare_reference
    ref_y, _ = prepare_reference(ref_y, par_y)
    ref, par = analyze_cached(ref_y, ref_t), analyze_cached(par_y, par_t or ref_t)
    res = compare(ref, par, par_y)
    if not 0 <= index < len(res["regions"]):
        raise HTTPException(404, "No such finding.")
    try:
        fixed, info = repair(par_y, ref, par, res["regions"][index].to_dict(), res["global"], ref_y)
    except ValueError as e:
        raise HTTPException(422, str(e))
    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = f"fix_{report_id}_{index}.wav"
    audio.save(UPLOADS / name, snippet(fixed, info["fixed"]))
    return {"edit": info["edit"], "url": f"/media/uploads/{name}", "span": info["fixed"]}


@app.post("/api/practice")
async def practice(report_id: str = Form(...), index: int = Form(...), retake: UploadFile = File(...)):
    """Practice loop: the participant re-reads just the passage of one finding.
    The original passage and the retake are each compared with the same crop
    of the reference, with identical processing, so before and after are
    directly comparable."""
    from .contrast import compare, pair_words
    from .pipeline import analyze_cached, prepare_reference
    ctx = _sample_ctx(report_id[2:]) if report_id.startswith("s-") else CTX.get(report_id)
    if ctx is None:
        raise HTTPException(404, "Analysis expired - analyse the recording again.")
    ref_y, ref_t, par_y, par_t = ctx
    ref_y, _ = prepare_reference(ref_y, par_y)
    ref, par = analyze_cached(ref_y, ref_t), analyze_cached(par_y, par_t or ref_t)
    res = compare(ref, par, par_y)
    if not 0 <= index < len(res["regions"]) or res["regions"][index].scope != "local":
        raise HTTPException(404, "No such passage.")
    g = res["regions"][index]
    pairs = pair_words(ref, par)
    p2r = {j: i for i, j in pairs}
    js = [j for j in range(g.word_start, g.word_end + 1) if j in p2r] or [min(p2r, key=lambda j: abs(j - g.word_start))]
    i0, i1 = p2r[js[0]], p2r[js[-1]]
    # widen to a speakable passage: at least 8 words, ending at punctuation when close
    while i1 - i0 + 1 < 8 and (i0 > 0 or i1 < len(ref.words) - 1):
        if i0 > 0:
            i0 -= 1
        if i1 - i0 + 1 < 8 and i1 < len(ref.words) - 1:
            i1 += 1
    for _ in range(4):
        if ref.words[i1].punct_after or i1 >= len(ref.words) - 1:
            break
        i1 += 1
    text = " ".join(w.text for w in ref.words[i0:i1 + 1])
    r2p = {i: j for i, j in pairs}

    def crop(y, a, b):
        return y[max(0, int((a - 0.3) * audio.SR)): int((b + 0.3) * audio.SR)]
    ref_c = crop(ref_y, ref.words[i0].start, ref.words[i1].end)
    j0, j1 = r2p.get(i0, js[0]), r2p.get(i1, js[-1])
    par_c = crop(par_y, par.words[j0].start, par.words[j1].end)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    new_y = audio.load(await _save(retake))
    if len(new_y) < audio.SR:
        raise HTTPException(422, "The retake is too short.")
    rc = analyze_cached(ref_c, text)

    def side(y):
        an = analyze_cached(y, text)
        try:
            out = compare(rc, an, y)
        except ValueError as e:
            raise HTTPException(422, f"Could not match the retake to the passage: {e}")
        locs = [r for r in out["regions"] if r.scope == "local"]
        same = [r for r in out["regions"] if r.type == g.type]
        fin = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 2)  # noqa: E731
        gp, gt = out["global"]["pace"], out["global"]["pitch"]
        return {"findings": [{"type": r.type, "severity": r.severity, "start": round(r.start, 2), "end": round(r.end, 2)}
                             for r in locs],
                "target_severity": max((r.severity for r in same), default=0),
                "pace_sps": fin(gp["participant_sps"]), "ref_pace_sps": fin(gp["reference_sps"]),
                "pitch_range_st": fin(gt["participant_range_st"]),
                "ref_pitch_range_st": fin(gt["reference_range_st"]),
                "duration": round(len(y) / audio.SR, 2)}
    before, after = side(par_c), side(new_y)
    tb, ta = before["target_severity"], after["target_severity"]
    if ta == 0 and tb > 0:
        verdict = "fixed"
    elif ta == 0:
        verdict = "clean"
    elif ta < tb:
        verdict = "improved"
    else:
        verdict = "still_there"
    name = f"practice_{report_id}_{index}.wav"
    audio.save(UPLOADS / name, new_y)
    return {"passage": text, "type": g.type, "verdict": verdict, "before": before, "after": after,
            "retake_url": f"/media/uploads/{name}"}


_PRECOMP_CACHE: dict | None = None


def _precomp() -> dict:
    global _PRECOMP_CACHE
    if _PRECOMP_CACHE is None:
        _PRECOMP_CACHE = {}
        for f in sorted(PRECOMP.glob("*.json")) if PRECOMP.exists() else []:
            _PRECOMP_CACHE.update(json.loads(f.read_text()))
    return _PRECOMP_CACHE


def _precomputed(sample_id: str, rubric: str) -> dict:
    """Dataset clips analysed ahead of time (tools/make_space.py): the report is
    stored once and only the rubric score is recomputed, so samples open
    instantly on a small server. Uploads are always analysed live."""
    from types import SimpleNamespace

    from .contrast import DIMENSION
    from .explain import load_rubric, score

    rep = json.loads(json.dumps(_precomp()[sample_id]))  # deep copy
    regs = [SimpleNamespace(**{k: r[k] for k in ("type", "scope", "severity", "word_start", "word_end")},
                            dimension=DIMENSION[r["type"]]) for r in rep["regions"]]
    rep["score"] = score(regs, len(rep["words"]), rep["duration"], load_rubric(rubric))
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
