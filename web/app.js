// SpeechLens dashboard - no build step, no external requests.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const NAMES = {
  rushed: "Rushed pace", dragging: "Dragging pace", monotone: "Monotone pitch", trailing_off: "Trailing off",
  pause_omission: "Missing pauses", awkward_pause: "Awkward pause", filler: "Filler sound",
  mumbling: "Muffled articulation", stutter: "Repeated onset",
};
const DIM_OF = {
  rushed: "pace", dragging: "pace", pause_omission: "pausing", awkward_pause: "pausing", filler: "fluency",
  stutter: "fluency", monotone: "pitch_variety", trailing_off: "volume", mumbling: "clarity",
};
const METRIC_NAMES = {
  articulation_rate: "Articulation rate", overall_articulation_rate: "Overall articulation rate",
  pitch_range_p10_p90: "Pitch range (p10–p90)", overall_pitch_range: "Overall pitch range",
  loudness_vs_reference: "Loudness", deepest_drop: "Deepest drop", energy_above_2kHz: "Energy above 2 kHz",
  energy_above_4kHz: "Energy above 4 kHz", recognizer_confidence: "Recognizer confidence",
  mean_pause_at_punctuation: "Pause at punctuation", pauses_lost: "Pauses affected", pause_length: "Pause length",
  voiced_sound_in_pause: "Voiced sound in pause", onset_repetitions: "Onset repetitions",
  burst_vs_word_onset_similarity: "Burst vs onset similarity",
  fragment_vs_word_onset_similarity: "Fragment vs word-onset similarity", held_steady_voicing: "Held steady vowel",
  time_added: "Time added at this boundary",
};
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const col = (t) => css(`--${DIM_OF[t] || t}`);
const fmtT = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

const S = { rubrics: [], library: [], manifest: { clips: [] }, rep: null, pps: 60, sel: -1, sample: null };
const audioEl = $("#audio");
const refAudio = $("#ref-audio");

// --------------------------------------------------------------- routing --
function route() {
  const v = (location.hash || "#analyze").slice(1);
  $$(".view").forEach((s) => (s.hidden = s.id !== `view-${v}`));
  $$(".tabs a").forEach((a) => a.setAttribute("aria-selected", a.dataset.view === v));
  if (v === "evaluation") renderEval();
  if (S.rep && v === "analyze") requestAnimationFrame(drawTape);
}
addEventListener("hashchange", route);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

// ------------------------------------------------------------------ boot --
async function boot() {
  route();
  const [rubrics, library, manifest, status] = await Promise.all([
    api("/api/rubrics"), api("/api/library"), api("/api/samples"), api("/api/status"),
  ]);
  Object.assign(S, { rubrics, library, manifest });
  $("#rubric").innerHTML = rubrics.map((r) => `<option value="${r.id}">${esc(r.name)}</option>`).join("");
  $("#ref-select").innerHTML = library.length
    ? library.map((l) => `<option value="${l.id}">${esc(l.title)}${l.speaker && l.speaker !== l.title ? ` (${esc(l.speaker)})` : ""}</option>`).join("")
    : `<option value="">No reference speeches yet</option>`;
  renderSamples();
  renderDataset();
  if (status.aligner !== "wav2vec2-ctc") setStatus("Alignment model not found: using the approximate fallback aligner.", true);
}

function setStatus(msg, err = false) {
  const el = $("#status");
  el.textContent = msg;
  el.classList.toggle("err", err);
}

// --------------------------------------------------------------- samples --
function titleOf(id) { return (S.library.find((l) => l.id === id) || {}).title || id; }

function clipCaption(c) {
  if (c.kind === "control") return { b: "Unflawed control", s: c.control.replace(/_/g, " ") };
  if (c.kind === "human") return { b: `Team recording: ${c.speaker[0].toUpperCase() + c.speaker.slice(1)}, take ${c.take}`, s: c.labels.map((l) => NAMES[l.type]).join(", ") + " (vs own clean take)" };
  if (c.kind === "human_reference") return { b: `Clean take: ${c.speaker}`, s: "reference for the takes above" };
  if (c.kind === "mixed") return { b: `Mixed flaws, level ${c.mix_level} of 8`, s: c.labels.map((l) => NAMES[l.type]).join(", ") };
  const l = c.labels[0];
  return { b: `${NAMES[l.type]}, severity ${l.severity}`, s: `${l.start.toFixed(1)}–${l.end.toFixed(1)} s` };
}

function renderSamples() {
  const pick = [];
  const bySrc = {};
  for (const c of S.manifest.clips) (bySrc[c.source] ||= []).push(c);
  for (const [src, cs] of Object.entries(bySrc)) {
    const want = cs.filter((c) => c.kind === "human" && c.speaker === "manya" && c.labels.some((l) => l.performed))
      .concat(cs.filter((c) => c.kind === "mixed" && [4, 6, 8].includes(c.mix_level) && !/_r\d/.test(c.id)))
      .concat(cs.filter((c) => c.kind === "single" && c.labels[0].severity === 4 && ["rushed", "monotone", "filler", "trailing_off"].includes(c.labels[0].type)));
    pick.push([src, want.slice(0, 8)]);
  }
  $("#sample-list").innerHTML = pick.map(([src, cs]) => `<div class="sample-group">${esc(titleOf(src))}</div>` +
    cs.map((c) => {
      const cap = clipCaption(c);
      const dots = c.labels.slice(0, 6).map((l) => `<i class="dot" style="background:${col(l.type)}"></i>`).join("");
      return `<button class="sample" data-id="${c.id}"><b>${esc(cap.b)}</b><span class="dots">${dots}</span><span>${esc(cap.s)}</span></button>`;
    }).join("")).join("") || `<p class="hint">Build the dataset to see clips here.</p>`;
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-id]");
  if (b && (b.classList.contains("sample") || b.dataset.act === "analyze")) {
    location.hash = "#analyze";
    analyzeSample(b.dataset.id);
  }
});

async function analyzeSample(id) {
  S.sample = id;
  const clip = S.manifest.clips.find((c) => c.id === id);
  if (clip && !clip.reference_clip) $("#ref-select").value = clip.source;
  $$(".sample").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  const fd = new FormData();
  fd.append("sample_id", id);
  fd.append("rubric", $("#rubric").value);
  await run(fd, `Analyzing ${id}…`);
}

// ------------------------------------------------------------------ form --
const parFile = $("#par-file");
parFile.addEventListener("change", () => ($("#par-name").textContent = parFile.files[0]?.name || "WAV, MP3, M4A, OGG"));
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) { parFile.files = e.dataTransfer.files; parFile.dispatchEvent(new Event("change")); }
});

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!parFile.files[0]) return setStatus("Choose the participant recording first.", true);
  S.sample = null;
  $$(".sample").forEach((b) => b.classList.remove("active"));
  const fd = new FormData();
  fd.append("rubric", $("#rubric").value);
  fd.append("participant_file", parFile.files[0]);
  if ($("#par-text").value.trim()) fd.append("participant_text", $("#par-text").value.trim());
  const rf = $("#ref-file").files[0];
  if (rf) {
    if (!$("#ref-text").value.trim()) return setStatus("Paste the transcript for your reference recording.", true);
    fd.append("reference_file", rf);
    fd.append("reference_text", $("#ref-text").value.trim());
  } else {
    if (!$("#ref-select").value) return setStatus("Choose a reference speech.", true);
    fd.append("reference_id", $("#ref-select").value);
  }
  await run(fd, "Aligning words and extracting features…");
});

$("#rubric").addEventListener("change", () => {
  if (S.sample) analyzeSample(S.sample);
  else if (S.rep) $("#form").requestSubmit();
});

async function run(fd, msg) {
  $("#go").disabled = true;
  setStatus(msg);
  try {
    const rep = await api("/api/analyze", { method: "POST", body: fd });
    S.rep = rep;
    S.sel = -1;
    showReport();
    const clip = S.sample && S.manifest.clips.find((c) => c.id === S.sample);
    const vs = clip && clip.reference_clip ? ` Compared with ${clip.speaker[0].toUpperCase() + clip.speaker.slice(1)}'s own clean take.` : "";
    setStatus(`Done. ${rep.regions.length} finding${rep.regions.length === 1 ? "" : "s"} in ${rep.duration.toFixed(1)} s of speech.${vs}`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    $("#go").disabled = false;
  }
}

// ---------------------------------------------------------------- report --
function showReport() {
  const r = S.rep;
  $("#empty").hidden = true;
  $("#report").hidden = false;
  audioEl.src = r.audio_url;
  refAudio.src = r.reference_url || "";
  $("#score-num").textContent = Math.round(r.score.overall);
  $("#score-band").textContent = r.score.band;
  $("#score-rubric").textContent = `${r.score.rubric} rubric`;
  $("#dims").innerHTML = Object.entries(r.score.dimensions).map(([d, v]) =>
    `<div class="dim" style="--c:var(--${d})"><span class="dim-name">${esc(r.dimension_labels[d])}</span>
     <span class="dim-val">${v.toFixed(1)}</span><span class="dim-bar"><i style="width:0"></i></span></div>`).join("");
  requestAnimationFrame(() => $$(".dim-bar i").forEach((el, i) => (el.style.width = `${Object.values(r.score.dimensions)[i] * 10}%`)));
  const hasTruth = Array.isArray(r.truth);
  $("#legend-truth").hidden = $("#cv-truth").hidden = $("#ll-truth").hidden = !hasTruth;
  const lanes = $("#lanes");
  S.pps = Math.max(4, (lanes.clientWidth || 800) / r.duration);  // open showing the whole recording
  drawTape();
  renderFindings();
}

function canvasFor(id, w) {
  const cv = $(id);
  const dpr = devicePixelRatio || 1;
  const h = cv.clientHeight;
  cv.style.width = `${w}px`;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}

function drawLine(g, t, ys, toY, color, dashed, pps) {
  g.strokeStyle = color;
  g.lineWidth = dashed ? 1.4 : 1.8;
  g.setLineDash(dashed ? [4, 3] : []);
  g.beginPath();
  let pen = false;
  for (let i = 0; i < t.length; i++) {
    const v = ys[i];
    if (v === null) { pen = false; continue; }
    const x = t[i] * pps, y = toY(v);
    pen ? g.lineTo(x, y) : g.moveTo(x, y);
    pen = true;
  }
  g.stroke();
  g.setLineDash([]);
}

function grid(g, w, h, vals, toY) {
  g.strokeStyle = css("--soft");
  g.fillStyle = css("--muted");
  g.font = "10px Archivo, sans-serif";
  g.lineWidth = 1;
  for (const v of vals) {
    const y = Math.round(toY(v)) + 0.5;
    g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
  }
}

function drawTape() {
  const r = S.rep;
  if (!r) return;
  const pps = S.pps;
  const W = Math.max($("#lanes").clientWidth, Math.ceil(r.duration * pps));
  const ink = css("--ink"), ref = css("--ref");
  // waveform
  {
    const [g, w, h] = canvasFor("#cv-wave", W);
    g.fillStyle = css("--muted");
    const step = 1 / 100;
    for (let i = 0; i < r.peaks.length; i++) {
      const a = r.peaks[i] * (h / 2 - 4);
      g.fillRect(i * step * pps, h / 2 - a, Math.max(1, step * pps - 0.3), a * 2 || 1);
    }
  }
  const t = r.series.t;
  const lane = (id, lo, hi, key, ticks) => {
    const [g, w, h] = canvasFor(id, W);
    const toY = (v) => h - 6 - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (h - 12);
    grid(g, w, h, ticks, toY);
    drawLine(g, t, r.series.reference[key], toY, ref, true, pps);
    drawLine(g, t, r.series.participant[key], toY, ink, false, pps);
  };
  lane("#cv-pitch", -12, 12, "f0_st", [-6, 0, 6]);
  lane("#cv-loud", -30, 10, "intensity_db", [-20, 0]);
  const artMax = Math.max(20, ...r.series.reference.hiband.filter((v) => v !== null), ...r.series.participant.hiband.filter((v) => v !== null));
  lane("#cv-art", 0, Math.min(80, artMax * 1.05), "hiband", [10, 20, 30]);
  // ground truth
  if (Array.isArray(r.truth)) {
    const [g, w, h] = canvasFor("#cv-truth", W);
    for (const l of r.truth) {
      const x0 = l.start * pps, x1 = Math.max(x0 + 3, l.end * pps);
      g.fillStyle = col(l.type);
      g.globalAlpha = l.performed === false ? 0.25 : 0.9;  // scripted but not performed: faint
      g.fillRect(x0, 4, x1 - x0, h - 8);
      g.globalAlpha = 1;
    }
  }
  // words
  const flagged = new Set();
  r.regions.forEach((g) => { if (g.scope === "local") for (let i = g.word_start; i <= g.word_end; i++) flagged.add(i); });
  const wordsEl = $("#words");
  wordsEl.style.width = `${W}px`;
  wordsEl.innerHTML = r.words.map((w) => {
    const x = w.start * pps, wd = Math.max(2, (w.end - w.start) * pps);
    const text = wd > 16 ? esc(w.text) : "";
    return `<span class="word${flagged.has(w.i) ? " flag" : ""}" data-t="${w.start}" title="${esc(w.text)}  ${w.start.toFixed(2)}–${w.end.toFixed(2)} s" style="left:${x}px;width:${wd}px">${text}</span>`;
  }).join("");
  // bands
  const bands = $("#bands");
  bands.style.width = `${W}px`;
  bands.innerHTML = r.regions.map((g, i) => {
    const c = `var(--${g.dimension})`;
    if (g.scope === "global") return `<div class="band global" data-i="${i}" style="--c:${c};left:0;width:${W}px"><span class="tag">${NAMES[g.type]}: whole reading</span></div>`;
    const x = g.start * pps, wd = Math.max(4, (g.end - g.start) * pps);
    return `<div class="band${i === S.sel ? " sel" : ""}" data-i="${i}" style="--c:${c};left:${x}px;width:${wd}px">${wd > 46 ? `<span class="tag">${NAMES[g.type]}</span>` : ""}</div>`;
  }).join("");
  placeHead();
}

$("#bands").addEventListener("click", (e) => {
  const b = e.target.closest(".band");
  if (b) { select(+b.dataset.i, true); e.stopPropagation(); }
});
$("#lanes").addEventListener("click", (e) => {
  if (e.target.closest(".band")) return;
  const lanes = $("#lanes");
  const x = e.clientX - lanes.getBoundingClientRect().left + lanes.scrollLeft;
  audioEl.currentTime = Math.max(0, x / S.pps);
  placeHead();
});
$("#zoom-in").onclick = () => zoom(1.6);
$("#zoom-out").onclick = () => zoom(1 / 1.6);
function zoom(f) {
  const lanes = $("#lanes");
  const center = (lanes.scrollLeft + lanes.clientWidth / 2) / S.pps;
  const fit = lanes.clientWidth / S.rep.duration;
  S.pps = Math.min(600, Math.max(fit, S.pps * f));
  drawTape();
  lanes.scrollLeft = center * S.pps - lanes.clientWidth / 2;
}
addEventListener("resize", () => { if (S.rep) drawTape(); });

// ---------------------------------------------------------------- player --
let stopAt = null, activeEl = audioEl;
$("#play").onclick = () => {
  stopAt = null;
  if (audioEl.paused) { refAudio.pause(); activeEl = audioEl; audioEl.play(); } else audioEl.pause();
};
audioEl.addEventListener("play", () => $("#play").classList.add("on"));
audioEl.addEventListener("pause", () => $("#play").classList.remove("on"));
function tick() {
  if (stopAt !== null && activeEl.currentTime >= stopAt) { activeEl.pause(); stopAt = null; }
  placeHead();
  requestAnimationFrame(tick);
}
function placeHead() {
  if (!S.rep) return;
  const t = audioEl.currentTime;
  const x = t * S.pps;
  $("#playhead").style.transform = `translateX(${x}px)`;
  $("#clock").textContent = fmtT(t);
  const lanes = $("#lanes");
  if (!audioEl.paused && (x < lanes.scrollLeft || x > lanes.scrollLeft + lanes.clientWidth - 40)) lanes.scrollLeft = x - 80;
  const cur = S.rep.words.find((w) => t >= w.start && t <= w.end);
  $$(".word.now").forEach((el) => el.classList.remove("now"));
  if (cur && !audioEl.paused) $(`.word[data-t="${cur.start}"]`)?.classList.add("now");
}
requestAnimationFrame(tick);
$("#words").addEventListener("click", (e) => {
  const w = e.target.closest(".word");
  if (w) { audioEl.currentTime = +w.dataset.t; placeHead(); e.stopPropagation(); }
});
function playSpan(el, a, b) {
  (el === audioEl ? refAudio : audioEl).pause();
  activeEl = el;
  el.currentTime = Math.max(0, a);
  stopAt = b;
  el.play();
}

// -------------------------------------------------------------- findings --
function pips(n) { return `<span class="pips" aria-label="severity ${n} of 5">${[1, 2, 3, 4, 5].map((i) => `<i class="${i <= n ? "on" : ""}"></i>`).join("")}</span>`; }

function renderFindings() {
  const r = S.rep;
  $("#flist-title").textContent = r.regions.length ? `Findings (${r.regions.length})` : "Findings";
  $("#flist").innerHTML = r.regions.length ? r.regions.map((g, i) => {
    const m = g.metrics[0];
    const when = g.scope === "global" ? "Whole reading" : `${g.start.toFixed(2)}–${g.end.toFixed(2)} s`;
    return `<li class="fi" data-i="${i}" style="--c:var(--${g.dimension})" tabindex="0">
      <span class="bar"></span><span class="t">${NAMES[g.type]}${pips(g.severity)}</span><span class="when">${when}</span>
      <span class="sub">${esc(g.text ? `“${g.text}”` : METRIC_NAMES[m.name] || m.name)}</span></li>`;
  }).join("") : `<li class="ok-msg">No delivery flaws found against this reference.</li>`;
  $("#fdetail").innerHTML = "";
  if (r.regions.length) select(0, false);
}
$("#flist").addEventListener("click", (e) => { const li = e.target.closest(".fi"); if (li) select(+li.dataset.i, true); });
$("#flist").addEventListener("keydown", (e) => { const li = e.target.closest(".fi"); if (li && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); select(+li.dataset.i, true); } });

const fmtV = (v, u) => {
  if (v === null || v === undefined) return "–";
  if (u === "s") return Math.abs(v) < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(2)} s`;
  if (u === "count") return `${Math.round(v)}`;
  if (u === "%") return `${v.toFixed(1)}%`;
  if (u === "cosine" || u === "") return v.toFixed(2);
  return `${v.toFixed(2)} ${u === "semitones" ? "st" : u}`;
};

function select(i, scroll) {
  S.sel = i;
  const g = S.rep.regions[i];
  $$(".fi").forEach((li) => li.classList.toggle("sel", +li.dataset.i === i));
  $$(".band").forEach((b) => b.classList.toggle("sel", +b.dataset.i === i));
  const [lead, why, fix] = splitExplanation(g.explanation);
  const truthHit = Array.isArray(S.rep.truth) ? S.rep.truth.find((l) => l.type === g.type && Math.min(l.end, g.end) - Math.max(l.start, g.start) > 0) : null;
  $("#fdetail").style.setProperty("--c", `var(--${g.dimension})`);
  $("#fdetail").innerHTML = `
    <h3>${NAMES[g.type]}${pips(g.severity)}</h3>
    <div class="meta">${g.scope === "global" ? "Whole reading" : `${g.start.toFixed(2)}–${g.end.toFixed(2)} s`}, severity ${g.severity} of 5, confidence ${Math.round(g.confidence * 100)}%${Array.isArray(S.rep.truth) ? (truthHit ? `. Matches injected ${NAMES[truthHit.type].toLowerCase()} at ${truthHit.start.toFixed(2)}–${truthHit.end.toFixed(2)} s (severity ${truthHit.severity})` : ". No matching ground-truth label") : ""}</div>
    <div class="abtn">
      ${g.scope === "local" ? `<button data-play="par">Play this part</button>` : ""}
      ${g.scope === "local" && S.rep.reference_url ? `<button data-play="ref">Play the reference here</button>` : ""}
    </div>
    <p>${esc(lead)}</p><p>${esc(why)}</p>${fix ? `<p class="fix">${esc(fix)}</p>` : ""}
    <table class="mt"><thead><tr><th>Measure</th><th>Reference</th><th>Participant</th><th>Change</th><th>z</th></tr></thead><tbody>
    ${g.metrics.map((m) => `<tr><td>${esc(METRIC_NAMES[m.name] || m.name)}</td><td>${fmtV(m.reference, m.unit)}</td><td>${fmtV(m.participant, m.unit)}</td>
      <td>${m.delta > 0 ? "+" : ""}${fmtV(m.delta, m.unit)}</td><td>${m.z === null ? "–" : m.z.toFixed(1)}</td></tr>
      ${m.threshold ? `<tr><td class="th" colspan="5">Rule: ${esc(m.threshold)}</td></tr>` : ""}`).join("")}
    </tbody></table>`;
  $$("#fdetail [data-play]").forEach((b) => (b.onclick = () => b.dataset.play === "par"
    ? playSpan(audioEl, g.start - 0.35, g.end + 0.35)
    : playSpan(refAudio, g.ref_start - 0.35, g.ref_end + 0.35)));
  if (scroll && g.scope === "local") {
    const lanes = $("#lanes");
    const x = g.start * S.pps;
    if (x < lanes.scrollLeft || x > lanes.scrollLeft + lanes.clientWidth - 60) lanes.scrollTo({ left: x - 60, behavior: "smooth" });
    audioEl.currentTime = g.start;
  }
}

function splitExplanation(s) {
  const fixAt = s.indexOf(" Fix: ");
  const body = fixAt >= 0 ? s.slice(0, fixAt) : s;
  const fix = fixAt >= 0 ? s.slice(fixAt + 1) : "";
  const parts = body.match(/[^.]+(?:\.\d+[^.]*)*\.(?=\s|$)/g) || [body];
  return [parts[0].trim(), parts.slice(1).join(" ").trim(), fix];
}

// --------------------------------------------------------------- dataset --
function renderDataset() {
  const lib = S.library;
  const clips = S.manifest.clips;
  $("#ds-sources").innerHTML = lib.map((l) => `<div class="src"><b>${esc(l.title)}</b>
    <span>${esc([l.speaker, l.year].filter(Boolean).join(", "))}</span>
    <span>${l.duration.toFixed(0)} s excerpt, ${l.words} words, median F0 ${l.f0_median_hz} Hz</span>
    <span>${clips.filter((c) => c.source === l.id).length} clips</span></div>`).join("");
  $("#f-src").innerHTML = `<option value="">All</option>` + lib.map((l) => `<option value="${l.id}">${esc(l.title)}</option>`).join("");
  $("#f-type").innerHTML = `<option value="">All</option>` + Object.entries(NAMES).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  ["#f-src", "#f-kind", "#f-type"].forEach((s) => ($(s).onchange = fillTable));
  fillTable();
}
function fillTable() {
  const src = $("#f-src").value, kind = $("#f-kind").value, type = $("#f-type").value;
  const rows = S.manifest.clips.filter((c) => (!src || c.source === src) && (!kind || c.kind === kind) && (!type || c.labels.some((l) => l.type === type)));
  $("#ds-count").textContent = `${rows.length} clips`;
  $("#ds-body").innerHTML = rows.slice(0, 400).map((c) => `<tr><td>${esc(c.id)}</td><td>${esc(c.kind)}</td>
    <td>${c.labels.length ? c.labels.map((l) => `<span class="lab"><i class="dot" style="background:${col(l.type)}"></i>${NAMES[l.type]} ${l.severity}, ${l.start.toFixed(1)}–${l.end.toFixed(1)} s</span>`).join("") : `<span class="hint">${esc(c.control ? c.control.replace(/_/g, " ") : "none")}</span>`}</td>
    <td>${c.duration.toFixed(1)} s</td><td><button data-act="analyze" data-id="${c.id}">Analyze</button></td></tr>`).join("");
}

// ------------------------------------------------------------ evaluation --
let evalDone = false;
async function renderEval() {
  if (evalDone) return;
  const e = await api("/api/evaluation");
  const el = $("#eval");
  if (!e.available) { el.innerHTML = `<h1>Evaluation</h1><p class="lede">No evaluation run yet. Run <code>python tools/evaluate.py</code>.</p>`; return; }
  evalDone = true;
  const g3 = e["grounding_tiou_0.3"], g5 = e["grounding_tiou_0.5"];
  const pct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);
  const sv = e.score_validity;
  const ctrlRows = Object.entries(e.controls);
  const fpm = ctrlRows.reduce((a, [, c]) => a + c.false_regions, 0) / Math.max(1e-9, ctrlRows.reduce((a, [, c]) => a + c.minutes, 0));
  const rowsTbl = (g) => Object.entries(g).map(([t, r]) => `<tr><td>${t === "__all__" ? "<b>All flaws</b>" : NAMES[t]}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.f1)}</td><td>${r.mean_tiou ?? "–"}</td><td>${r.boundary_err_ms ?? "–"}</td><td>${r.tp}/${r.tp + r.fn}</td></tr>`).join("");
  const heat = (v) => v === null ? "" : `background:color-mix(in srgb, var(--accent) ${Math.round(v * 70)}%, transparent)`;
  const cv = e.cross_validated || null;
  const syn = cv ? cv.cv_synthetic : g3;
  const hum = cv ? cv.cv_human : null;
  const mc = e.manipulation_check;
  const simple = (g) => Object.entries(g).map(([t, r]) => `<tr><td>${t === "__all__" ? "<b>All flaws</b>" : NAMES[t]}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.f1)}</td><td>${r.tp}/${r.tp + r.fn}</td></tr>`).join("");
  el.innerHTML = `
    <h1>How well it works</h1>
    <p class="lede">Every number comes from the full pipeline on raw audio, including forced alignment of the flawed recording. Detection thresholds are fitted from data, so each result below is cross-validated: synthetic clips are scored with thresholds fitted without that speech, and human recordings with thresholds fitted without that recorder.</p>
    <div class="kpis">
      <div class="kpi"><b>${pct(syn.__all__.f1)}</b><span>F1 locating synthetic flaws (precision ${pct(syn.__all__.precision)}), leave-one-speech-out</span></div>
      ${hum ? `<div class="kpi"><b>${pct(hum.__all__.recall)}</b><span>of verified human flaws found (precision ${pct(hum.__all__.precision)} against scripted labels only), leave-one-speaker-out</span></div>` : ""}
      <div class="kpi"><b>${cv ? cv.cv_controls_false_per_min : fpm.toFixed(2)}</b><span>false findings per minute on unflawed controls, including the voice shifted ±4 semitones</span></div>
      <div class="kpi"><b>${sv.spearman_mix_level_vs_score ?? "–"}</b><span>Spearman ρ between flaw level and score (scores fall as delivery worsens)</span></div>
      <div class="kpi"><b>${e.reproducible ? "Identical" : "Differs"}</b><span>output on a re-run in a fresh process</span></div>
    </div>
    <h2>Synthetic flaws: cross-validated, tIoU ≥ 0.3</h2>
    <table class="et"><thead><tr><th>Flaw</th><th>Precision</th><th>Recall</th><th>F1</th><th>Found</th></tr></thead><tbody>${simple(syn)}</tbody></table>
    <p class="note">Temporal accuracy of matched flaws: mean tIoU ${g3.__all__.mean_tiou}, median boundary error ${g3.__all__.boundary_err_ms} ms.</p>
    ${hum ? `<h2>Team recordings: cross-validated, tIoU ≥ 0.3</h2>
    <p class="note">Three team members read three passages four times each: one clean take and three takes with scripted flaws. Each flawed take is compared with the same person's clean take. A manipulation check kept only flaws actually performed to at least severity-1 strength: ${mc ? `${mc.total.performed} of ${mc.total.scripted}` : "–"} were. Precision here counts any finding outside a scripted flaw as false, so unscripted slips the recorders made also count against it; a blind listening audit estimates the real rate.</p>
    <table class="et"><thead><tr><th>Flaw</th><th>Precision</th><th>Recall</th><th>F1</th><th>Found</th></tr></thead><tbody>${simple(hum)}</tbody></table>` : ""}
    ${mc ? `<h2>Manipulation check</h2><p class="note">How many scripted flaws each type's recordings actually contained, measured against the speaker's clean take with fixed criteria independent of the detector.</p>
    <table class="et"><thead><tr><th>Flaw</th><th>Performed</th><th>Criterion</th></tr></thead><tbody>${Object.entries(mc.performed).map(([t, v]) => `<tr><td>${NAMES[t]}</td><td>${v.performed}/${v.scripted}</td><td>${esc(mc.criteria[t].measure)} ${esc(mc.criteria[t].criterion)}</td></tr>`).join("")}</tbody></table>` : ""}
    ${e.audit ? `<h2>Blind listening audit</h2><p class="note">${esc(e.audit.summary)}</p>` : ""}
    <h2>Detection rate by severity</h2>
    <p class="note">Severity 1 is designed to be near-perfect and sits inside the natural variation of a person rereading a text, so it is mostly not reported; the rate should climb toward severity 5.</p>
    <table class="et heat"><thead><tr><th>Flaw</th>${[1, 2, 3, 4, 5].map((s) => `<th>Severity ${s}</th>`).join("")}</tr></thead><tbody>
      ${Object.entries(e.recall_by_severity).map(([t, row]) => `<tr><td>${NAMES[t]}</td>${row.map((v) => `<td class="h" style="${heat(v)}">${pct(v)}</td>`).join("")}</tr>`).join("")}</tbody></table>
    <h2>Severity estimate</h2>
    ${e.severity_calibration ? `<p class="note">Severity tables are fitted from data. On speeches held out of the fit: mean absolute error ${e.severity_calibration.calibrated_leave_one_speech_out.mae} levels, ${pct(e.severity_calibration.calibrated_leave_one_speech_out.within_1)} within one level (n = ${e.severity_calibration.calibrated_leave_one_speech_out.n}). With the injection engine's own parameter tables: ${e.severity_calibration.default_tables.mae} levels.</p>` : ""}
    <h2>Unflawed controls</h2>
    <p class="note">The same reference delivery, unchanged, with the voice shifted 4 semitones up or down, or 12 dB quieter. A speaker-agnostic system should report nothing here.</p>
    <table class="et"><thead><tr><th>Control</th><th>Clips</th><th>False findings</th><th>Per minute</th><th>Mean score</th></tr></thead><tbody>
      ${ctrlRows.map(([k, c]) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td>${c.clips}</td><td>${c.false_regions}</td><td>${c.false_per_min}</td><td>${c.mean_score}</td></tr>`).join("")}</tbody></table>
    <h2>Score follows delivery quality</h2>
    <p class="note">Mean overall score by mixed-flaw level (1 = two mild flaws, 8 = six severe flaws). Spearman ρ = ${sv.spearman_mix_level_vs_score ?? "–"}.</p>
    ${mixChart(sv.mean_score_by_mix_level)}`;
}

function mixChart(m) {
  const ks = Object.keys(m || {});
  if (!ks.length) return "";
  const w = 560, h = 180, bw = (w - 40) / ks.length;
  const bars = ks.map((k, i) => {
    const v = m[k], bh = (v / 100) * (h - 40);
    return `<rect x="${30 + i * bw + 6}" y="${h - 20 - bh}" width="${bw - 12}" height="${bh}" rx="3" fill="var(--accent)"/>
      <text x="${30 + i * bw + bw / 2}" y="${h - 24 - bh}" text-anchor="middle" font-size="11" fill="var(--ink)">${v}</text>
      <text x="${30 + i * bw + bw / 2}" y="${h - 5}" text-anchor="middle" font-size="11" fill="var(--muted)">${k}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px" role="img" aria-label="Mean score by mix level">${bars}</svg>`;
}

boot().catch((e) => setStatus(e.message, true));
