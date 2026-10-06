"""Build the hosted static demo into build/static/.

The same dashboard as the full app, served as plain files: dataset clips open
with reports pre-computed by the real pipeline (via the API, so they are
byte-for-byte what the server returns) and rubric scores are recomputed in the
browser. Requires build/space/ (tools/make_space.py) for the curated clip set.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

from speechlens.app import app  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "build/space"
OUT = ROOT / "build/static"


def mp3(src, dst):
    # the static host serves MP3 but not FLAC; 96 kbps mono keeps speech clean
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-b:a", "96k", str(dst)],
                   check=True)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)
    (OUT / "reports").mkdir()
    (OUT / "media/dataset").mkdir(parents=True)
    (OUT / "media/library").mkdir(parents=True)
    (OUT / "vendor").mkdir()
    client = TestClient(app)
    for route, name in (("/api/rubrics", "rubrics"), ("/api/library", "library"), ("/api/status", "status"),
                        ("/api/evaluation", "evaluation")):
        (OUT / "data" / f"{name}.json").write_text(json.dumps(client.get(route).json(), separators=(",", ":")))
    man = json.loads((SPACE / "data/dataset/manifest.json").read_text())
    man["note"] = "Hosted demo subset; the full dataset is linked from the README."
    (OUT / "data/samples.json").write_text(json.dumps(man, separators=(",", ":")))
    for c in man["clips"]:
        mp3(SPACE / "data/dataset/clips" / f"{c['id']}.flac", OUT / "media/dataset" / f"{c['id']}.mp3")
    for lib in (ROOT / "data/library").iterdir():
        if (lib / "audio.wav").exists():
            mp3(lib / "audio.wav", OUT / "media/library" / f"{lib.name}.mp3")
    n = 0
    for f in sorted((SPACE / "data/precomputed").glob("*.json")):
        for k, rep in json.loads(f.read_text()).items():
            ref = rep.get("reference_url") or ""
            m = re.match(r"/media/library/(\w+)/audio\.wav", ref)
            if m:
                rep["reference_url"] = f"media/library/{m.group(1)}.mp3"
            rep["audio_url"] = rep["audio_url"].lstrip("/").replace(".flac", ".mp3")
            if rep.get("reference_url"):
                rep["reference_url"] = rep["reference_url"].lstrip("/").replace(".flac", ".mp3")
            (OUT / "reports" / f"{k}.json").write_text(json.dumps(rep, separators=(",", ":")))
            n += 1
    shutil.copy(ROOT / "web/vendor/archivo.woff2", OUT / "vendor/archivo.woff2")
    # one self-contained page: styles and script inline
    html = (ROOT / "web/index.html").read_text()
    css = (ROOT / "web/app.css").read_text()
    js = (ROOT / "web/app.js").read_text()
    title = re.search(r"<title>.*?</title>", html).group(0)
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    body = re.sub(r'<script src="app.js" type="module"></script>', "", body)
    page = (f"<meta charset=\"utf-8\">\n{title}\n<style>\n{css}\n</style>\n{body}\n<script>window.SPEECHLENS_STATIC = true;</script>\n"
            f"<script type=\"module\">\n{js}\n</script>\n")
    (OUT / "index.html").write_text(page)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{n} reports, {len(man['clips'])} clips, {sum(1 for p in OUT.rglob('*') if p.is_file())} files, "
          f"{size / 1e6:.0f} MB -> {OUT}")


if __name__ == "__main__":
    main()
