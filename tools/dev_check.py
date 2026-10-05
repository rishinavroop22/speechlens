"""Dev check: inject each flaw into a clip, analyse with oracle word timings,
and print what the contrast engine finds."""
import sys
sys.path.insert(0, ".")
from speechlens import audio, features
from speechlens.contrast import compare
from speechlens.inject import Injector, FLAW_TYPES

wav, txt = sys.argv[1], open(sys.argv[2]).read()
sev = int(sys.argv[3]) if len(sys.argv) > 3 else 4
y = audio.load(wav)
ref = features.analyze(y, txt)
for ft in FLAW_TYPES:
    for seed in (1, 2):
        inj = Injector(y, ref.words, ref.ff.f0_median_hz, seed=seed)
        ref_o = features.Analysis(ref.duration, inj.words, features.word_features(inj.words, ref.ff), ref.ff, "oracle")
        out, nw, lb = inj.apply([(ft, sev)])
        ff = features.frame_features(out)
        par = features.Analysis(len(out) / 16000, nw, features.word_features(nw, ff), ff, "oracle")
        res = compare(ref_o, par)
        truth = [(l.start, l.end) for l in lb]
        found = [(r.type, r.start, r.end, r.severity) for r in res["regions"]]
        print(f"{ft:15s} s{seed} truth={truth} found={found}")
