"""Download the alignment model (wav2vec2-base-960h, ONNX) into models/wav2vec2/."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/Xenova/wav2vec2-base-960h/resolve/main/"
FILES = ["onnx/model.onnx", "vocab.json", "config.json", "preprocessor_config.json"]
DEST = Path(__file__).resolve().parent.parent / "models" / "wav2vec2"


def main():
    for f in FILES:
        out = DEST / f
        if out.exists() and out.stat().st_size > 0:
            print(f"ok   {f}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"get  {f} ...", flush=True)
        tmp = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(BASE + f, tmp)
        tmp.rename(out)
    print("model ready in", DEST)


if __name__ == "__main__":
    sys.exit(main())
