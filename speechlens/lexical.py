"""Reading errors: where the participant departs from the text itself.

Two flaw types, both contrastive against the reference reading of the same
transcript:

skipped_word  Forced alignment has to place every transcript word somewhere.
              A word that was never said is squeezed into a few frames with a
              near-zero acoustic posterior, while the reference says it
              clearly. Runs of such words are candidates; when the audio is
              available, a short recognition pass over the neighbourhood
              confirms the words are really absent (a mumbled word is still
              recognised, a skipped one is not).

restart       The reader goes back and says words again ("we choose to - we
              choose to go"). The aligner absorbs the extra audio into a long
              "pause" or a stretched word, i.e. added time at one boundary. At
              each such boundary we recognise the surrounding 3-5 s and look
              for an n-gram said twice in a row that the transcript does not
              repeat there.

The recogniser is the same wav2vec2 CTC model used for alignment, decoded
greedily with word timestamps, run only on short windows.
"""
from __future__ import annotations

import numpy as np

from .audio import SR

LOW_POSTERIOR = 0.10      # a skipped word's mean posterior of its own characters
REF_POSTERIOR = 0.50      # ... while the reference says it clearly
SQUEEZE = 0.55            # and its aligned duration shrinks below this share
RESTART_EXTRA_S = 0.30    # time added at a boundary before we listen for a restart


def recognize(y: np.ndarray, a: float, b: float) -> list[tuple[str, float, float]]:
    """Greedy CTC words with absolute timestamps for y[a:b] (seconds)."""
    from .align import FRAME, _session, emissions
    a = max(0.0, a)
    seg = y[int(a * SR): int(b * SR)]
    if len(seg) < int(0.3 * SR):
        return []
    _, vocab = _session()
    inv = {v: k for k, v in vocab.items()}
    blank = vocab.get("<pad>", 0)
    ids = emissions(seg).argmax(axis=1)
    words, cur, t0, prev = [], "", None, -1
    for t, i in enumerate(ids):
        if i != prev and i != blank:
            ch = inv.get(int(i), "")
            if ch == "|":
                if cur:
                    words.append((cur, a + t0 * FRAME, a + t * FRAME))
                cur, t0 = "", None
            elif ch not in ("<s>", "</s>", "<unk>"):
                if t0 is None:
                    t0 = t
                cur += ch
        prev = i
    if cur:
        words.append((cur, a + t0 * FRAME, a + len(ids) * FRAME))
    return words


def _norm(w: str) -> str:
    return "".join(ch for ch in w.upper() if ch.isalpha() or ch == "'")


def _same(a: str, b: str) -> bool:
    """Equal up to one character for words of 4+ letters (recogniser slips)."""
    a, b = _norm(a), _norm(b)
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
        return False
    import difflib
    return difflib.SequenceMatcher(a=a, b=b).ratio() >= 0.7


def _repeat(seq: list[str], maxn: int = 4, ok=lambda k, n: True):
    """First (k, n) with seq[k:k+n] == seq[k+n:k+2n], longest n first, that
    passes `ok`."""
    for n in range(maxn, 0, -1):
        for k in range(len(seq) - 2 * n + 1):
            if all(_same(seq[k + i], seq[k + n + i]) for i in range(n)):
                if n == 1 and len(_norm(seq[k])) < 3:
                    continue
                if ok(k, n):
                    return k, n
    return None


def detect(ref, par, par_y, pairs, raw_extra, Region, Metric, sevt) -> list:
    """Reading-error regions. `pairs`/`raw_extra` come from contrast.compare."""
    regions = []
    n = len(pairs)
    P = lambda j: pairs[j][1]  # noqa: E731

    # ---- skipped words --------------------------------------------------
    flag = np.zeros(n, bool)
    for j, (i, k) in enumerate(pairs):
        wp, wr = par.words[k], ref.words[i]
        dp, dr = wp.end - wp.start, wr.end - wr.start
        flag[j] = wp.conf < LOW_POSTERIOR and wr.conf >= REF_POSTERIOR and dp < SQUEEZE * dr
    j = 0
    while j < n:
        if not flag[j]:
            j += 1
            continue
        j0 = j
        while j + 1 < n and flag[j + 1]:
            j += 1
        j1 = j
        j += 1
        a, b = max(0, j0 - 1), min(n - 1, j1 + 1)
        missing = [par.words[P(q)].text for q in range(j0, j1 + 1)]
        if len(missing) == 1:
            # a lone short function word ("and", "a") after a pause is easily
            # mis-placed by the aligner; demand a much stronger squeeze
            q = P(j0)
            wr = ref.words[pairs[j0][0]]
            if len(par.words[q].norm) <= 3 and par.words[q].end - par.words[q].start > 0.35 * (wr.end - wr.start):
                continue
        s, e = par.words[P(a)].start, par.words[P(b)].end
        heard = None
        if par_y is not None:
            rec = recognize(par_y, s - 0.15, e + 0.15)
            heard = " ".join(w for w, _, _ in rec).lower()
            found = sum(any(_same(m, w) for w, _, _ in rec) for m in missing)
            if found * 2 >= len(missing) and len(missing) > 0:
                continue  # the words are there, only unclear: not a skip
        cp = float(np.mean([par.words[P(q)].conf for q in range(j0, j1 + 1)]))
        cr = float(np.mean([ref.words[pairs[q][0]].conf for q in range(j0, j1 + 1)]))
        reg = Region("skipped_word", round(s, 3), round(e, 3), P(a), P(b),
                     sevt("skipped_word", len(missing), [1, 1, 2, 2, 3]), round(min(0.95, 0.5 + 0.5 * (cr - cp)), 3),
                     [Metric("words_missing", "count", 0, len(missing), len(missing), None,
                             f"transcript word(s) with recogniser posterior < {LOW_POSTERIOR} (reference >= "
                             f"{REF_POSTERIOR}) squeezed below {int(SQUEEZE * 100)} % of the reference duration"),
                      Metric("recognizer_confidence", "", cr, cp, cp - cr)],
                     "local", ref.words[pairs[a][0]].start, ref.words[pairs[b][0]].end,
                     " ".join(w.text for w in par.words[P(a):P(b) + 1]))
        reg.evidence = (f"missing: “{' '.join(missing)}”" + (f"; recognised here: “{heard}”" if heard else ""))
        regions.append(reg)

    # ---- restarts -------------------------------------------------------
    if par_y is None:
        return regions
    cands = []
    for j in range(n - 1):
        k = P(j)
        dp = par.words[k].end - par.words[k].start
        dr = ref.words[pairs[j][0]].end - ref.words[pairs[j][0]].start
        if (np.isfinite(raw_extra[j]) and raw_extra[j] >= RESTART_EXTRA_S) or dp - dr >= RESTART_EXTRA_S + 0.05:
            cands.append(j)
    done_until = -1.0
    for j in cands:
        a, b = max(0, j - 4), min(n - 1, j + 4)
        s, e = par.words[P(a)].start, par.words[P(b)].end
        if s < done_until:
            continue
        rec = recognize(par_y, s, e)
        txt_all = [ref.words[pairs[q][0]].norm.replace("|", " ") for q in range(a, b + 1)]

        def real_words(k, m):
            # both copies must be words of this passage, said at word length -
            # a repeated onset fragment ("d- d- deliberately") is a stutter
            for c in (rec[k:k + m], rec[k + m:k + 2 * m]):
                if sum(t1 - t0 for _, t0, t1 in c) < 0.12 * m:
                    return False
            hits = sum(any(_same(w, t) for t in txt_all) for w, _, _ in rec[k:k + m])
            return hits * 2 >= m and any(len(_norm(w)) >= 4 and any(_same(w, t) for t in txt_all)
                                         for w, _, _ in rec[k:k + m])
        hit = _repeat([w for w, _, _ in rec], ok=real_words)
        if not hit:
            continue
        kk, m = hit
        rep_words = [w for w, _, _ in rec[kk:kk + m]]
        # natural repetition in the text itself ("we shall fight, we shall
        # fight") is in the reference too: not a restart
        txt = [ref.words[pairs[q][0]].norm.replace("|", " ") for q in range(a, b + 1)]
        tr = _repeat(txt)
        if tr and _same(txt[tr[0]], rep_words[0]):
            continue
        t_s, t_e = rec[kk + m - 1][2], rec[kk + 2 * m - 1][2]
        # nearest participant words for the index range
        wi = [q for q in range(P(a), P(b) + 1) if par.words[q].end > t_s - 0.05 and par.words[q].start < t_e + 0.05]
        w0, w1 = (wi[0], wi[-1]) if wi else (P(j), P(j + 1))
        heard = " ".join(w for w, _, _ in rec).lower()
        added = float(raw_extra[j]) if np.isfinite(raw_extra[j]) else 0.0
        reg = Region("restart", round(t_s, 3), round(t_e, 3), w0, w1,
                     sevt("restart", m, [1, 2, 2, 3, 4]), round(min(0.95, 0.55 + 0.1 * m), 3),
                     [Metric("words_repeated", "count", 0, m, m, None,
                             "the same word sequence recognised twice in a row, not repeated in the transcript"),
                      Metric("time_added", "s", 0, added, added)],
                     "local", None, None, " ".join(rep_words).lower())
        reg.ref_start = ref.words[pairs[a][0]].start
        reg.ref_end = ref.words[pairs[b][0]].end
        reg.evidence = f"recognised here: “{heard}”"
        regions.append(reg)
        done_until = t_e
    return regions


def vet_fillers(regions, ref, par, par_y, pairs) -> list:
    """A "filler" that the recogniser hears as a word of this passage is
    speech (usually a missed restart), not an "uh": drop it."""
    if par_y is None:
        return regions
    out = []
    for r in regions:
        if r.type == "filler" and r.scope == "local":
            rec = recognize(par_y, r.start - 0.05, r.end + 0.05)
            near = [ref.words[i].norm.replace("|", " ") for i, j in pairs
                    if r.word_start - 4 <= j <= r.word_end + 4]
            if any(len(_norm(w)) >= 4 and any(_same(w, t) for t in near) for w, _, _ in rec):
                continue
        out.append(r)
    return out
