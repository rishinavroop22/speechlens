"""Developer-only TTS (eSpeak NG via ctypes) to create test fixtures before real
speeches arrive. NOT used for the dataset itself."""
import ctypes, sys, numpy as np, soundfile as sf
import espeakng_loader as L

_lib = ctypes.CDLL(L.get_library_path())
_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
_buf = []

def _cb(wav, n, ev):
    if n > 0:
        _buf.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
    return 0
_cbf = _CB(_cb)
_sr = _lib.espeak_Initialize(1, 0, L.get_data_path().encode(), 0)
_lib.espeak_SetSynthCallback(_cbf)

def synth(text, out, rate=160, pitch=50, rng=60, voice=b"en-us"):
    _buf.clear()
    _lib.espeak_SetVoiceByName(voice)
    _lib.espeak_SetParameter(1, rate, 0)
    _lib.espeak_SetParameter(3, pitch, 0)
    _lib.espeak_SetParameter(4, rng, 0)
    t = text.encode()
    _lib.espeak_Synth(t, len(t) + 1, 0, 1, 0, 0, None, None)
    _lib.espeak_Synchronize()
    y = np.concatenate(_buf).astype(np.float32) / 32768
    sf.write(out, y, _sr)
    return out

if __name__ == "__main__":
    synth(sys.argv[1], sys.argv[2])
