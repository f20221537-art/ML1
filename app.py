import streamlit as st
import os, sys, subprocess, warnings, tempfile, shutil
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Instrument Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark navy theme matching the project)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

:root {
    --bg:        #0F172A;
    --surface:   #1A2745;
    --card:      #1E3258;
    --accent:    #00C2FF;
    --accent2:   #7C3AED;
    --green:     #00D98A;
    --orange:    #FF7C2A;
    --text:      #E2EAF4;
    --muted:     #7A90B0;
    --border:    #2A3F65;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4 { font-family: 'Space Mono', monospace; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff !important;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 14px;
    padding: 0.6rem 2rem;
    letter-spacing: 1px;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stFileUploader"] {
    background-color: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

.stProgress > div > div { background: var(--accent) !important; }

div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

.stAlert { border-radius: 10px !important; }

/* Instrument result cards */
.inst-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid var(--accent);
}
.inst-card h4 { margin: 0 0 0.4rem 0; font-family: 'Space Mono', monospace; font-size: 15px; }
.inst-card .ts  { font-size: 12px; color: var(--muted); margin: 2px 0; font-family: 'Space Mono', monospace; }
.inst-card .dur { font-size: 11px; color: var(--green); font-weight: 600; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1A2745 50%, #0F172A 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,194,255,0.12), transparent 70%);
}
.hero h1 { font-size: 2rem; margin: 0; color: #fff; letter-spacing: -1px; }
.hero p  { color: var(--muted); margin: 0.4rem 0 0 0; font-size: 15px; }
.badge {
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    border-radius: 6px;
    padding: 2px 10px;
    margin-right: 6px;
    margin-top: 10px;
}

/* Step tags */
.step-tag {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    font-size: 13px;
    color: var(--text);
}
.step-tag span { color: var(--accent); font-weight: 700; margin-right: 6px; }

/* Color helpers */
.c-cyan   { color: #00C2FF !important; }
.c-violet { color: #7C3AED !important; }
.c-green  { color: #00D98A !important; }
.c-orange { color: #FF7C2A !important; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS  (kept in sync with notebook)
# ─────────────────────────────────────────────
TARGET_SR          = 22050
WINDOW_DURATION    = 1.0
HOP_DURATION       = 0.5
CONFIDENCE_THRESHOLD = 0.10
MAX_SILENCE_GAP    = 2.0
MIN_PLAY_TIME      = 1.5
MODEL_PATH         = "instrument_model.keras"
LABELS_PATH        = "label_classes.npy"

INSTRUMENT_NAMES = {
    "cel": "Cello",        "clt": "Clarinet",
    "flu": "Flute",        "gac": "Acoustic Guitar",
    "gel": "Electric Guitar", "org": "Organ",
    "pia": "Piano",        "sax": "Saxophone",
    "tru": "Trumpet",      "vio": "Violin",
    "voi": "Voice / Vocals", "dru": "Drums",
}
INST_COLORS = {
    "cel":"#00C2FF","clt":"#7C3AED","flu":"#00D98A","gac":"#FF7C2A",
    "gel":"#00C2FF","org":"#7C3AED","pia":"#00D98A","sax":"#FF7C2A",
    "tru":"#00C2FF","vio":"#7C3AED","voi":"#00D98A","dru":"#FF7C2A",
}

# ─────────────────────────────────────────────
# ORIGINAL NOTEBOOK FUNCTIONS  (untouched)
# ─────────────────────────────────────────────

def extract_mel_spectrogram(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    return librosa.power_to_db(mel_spec, ref=np.max)


# ─────────────────────────────────────────────
# STREAMLIT HELPERS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    le = LabelEncoder()
    le.classes_ = np.load(LABELS_PATH, allow_pickle=True)
    return model, le


def run_demucs(audio_path: str, work_dir: str) -> str | None:
    """Run demucs on the given file; returns the stem folder path or None."""
    fname = os.path.basename(audio_path)
    result = subprocess.run(
        [sys.executable, "-m", "demucs", audio_path,
         "--out", os.path.join(work_dir, "separated")],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        st.error(f"Demucs failed:\n```\n{result.stderr[-2000:]}\n```")
        return None
    stem_folder = os.path.join(
        work_dir, "separated", "htdemucs",
        os.path.splitext(fname)[0]
    )
    return stem_folder


def scan_stems(stem_folder: str, model, label_encoder, progress_bar, status_text):
    stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    time_templates = []
    total_steps = len(stems)

    for stem_idx, stem in enumerate(stems):
        stem_path = os.path.join(stem_folder, stem)
        if not os.path.exists(stem_path):
            continue
        status_text.markdown(f"<p style='color:#7A90B0;font-size:13px;'>🔬 Scanning <b>{stem}</b>…</p>",
                             unsafe_allow_html=True)

        audio_data, sample_rate = librosa.load(stem_path, sr=TARGET_SR)
        window_samples = int(WINDOW_DURATION * sample_rate)
        hop_samples    = int(HOP_DURATION    * sample_rate)

        windows = list(range(0, len(audio_data) - window_samples, hop_samples))
        for w_idx, start in enumerate(windows):
            window = audio_data[start : start + window_samples]
            if len(window) < window_samples:
                window = np.pad(window, (0, window_samples - len(window)))

            spectrogram = extract_mel_spectrogram(window, sample_rate)
            if spectrogram.shape[1] != 44:
                continue

            probs = model.predict(spectrogram[np.newaxis, ..., np.newaxis], verbose=0)[0]
            for idx, confidence in enumerate(probs):
                if confidence > CONFIDENCE_THRESHOLD:
                    time_templates.append({
                        "start":      start / sample_rate,
                        "end":        (start + window_samples) / sample_rate,
                        "instrument": label_encoder.inverse_transform([idx])[0],
                    })

        progress_bar.progress((stem_idx + 1) / total_steps)

    return time_templates


def merge_timestamps(time_templates):
    instrument_groups = {}
    for p in time_templates:
        instrument_groups.setdefault(p["instrument"], []).append((p["start"], p["end"]))

    results = {}
    for inst, timings in instrument_groups.items():
        timings.sort(key=lambda x: x[0])
        blocks = []
        for start, end in timings:
            if not blocks:
                blocks.append([start, end])
            else:
                if start - blocks[-1][1] <= MAX_SILENCE_GAP:
                    blocks[-1][1] = max(blocks[-1][1], end)
                else:
                    blocks.append([start, end])
        filtered = [(s, e) for s, e in blocks if (e - s) >= MIN_PLAY_TIME]
        if filtered:
            results[inst] = filtered
    return results


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    conf_thresh = st.slider(
        "Confidence Threshold", 0.05, 0.50, CONFIDENCE_THRESHOLD, 0.05,
        help="Minimum model confidence (0–1) for an instrument to be logged."
    )
    silence_gap = st.slider(
        "Max Silence Gap (s)", 0.5, 5.0, MAX_SILENCE_GAP, 0.5,
        help="Merge two detections if the gap between them is ≤ this value."
    )
    min_play = st.slider(
        "Min Play Duration (s)", 0.5, 5.0, MIN_PLAY_TIME, 0.5,
        help="Ignore instrument blocks shorter than this duration."
    )

    st.markdown("---")
    st.markdown("### 📁 Model Files")

    model_ok  = os.path.exists(MODEL_PATH)
    labels_ok = os.path.exists(LABELS_PATH)

    st.markdown(
        f"{'✅' if model_ok  else '❌'} `instrument_model.keras`  \n"
        f"{'✅' if labels_ok else '❌'} `label_classes.npy`"
    )
    if not model_ok or not labels_ok:
        st.warning("Place model files in the app root before running.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "<p style='font-size:12px;color:#7A90B0;'>"
        "Trained on the <b>IRMAS</b> dataset. "
        "Uses <b>Demucs</b> for source separation and a <b>CNN</b> to classify instruments "
        "from Mel Spectrograms in 1-second sliding windows."
        "</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🎵 Instrument Classifier</h1>
  <p>Upload a song — the AI detects which instruments play and exactly when.</p>
  <span class='badge'>CNN</span>
  <span class='badge'>Mel Spectrogram</span>
  <span class='badge'>Demucs</span>
  <span class='badge'>IRMAS</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PIPELINE OVERVIEW (collapsible)
# ─────────────────────────────────────────────
with st.expander("📖 How it works", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "🎵", "1. Upload", "Drop a `.wav` file"),
        (c2, "✂️", "2. Separate", "Demucs splits it into vocals / drums / bass / other"),
        (c3, "🔬", "3. Analyse", "CNN scans each stem in 1-second sliding windows"),
        (c4, "📋", "4. Output",  "Timestamped instrument list with durations"),
    ]:
        col.markdown(f"**{icon} {title}**  \n{desc}")


# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your audio file here",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    label_visibility="collapsed"
)

if uploaded:
    st.audio(uploaded, format=uploaded.type)

    run_btn = st.button("🚀  Analyse Instruments", use_container_width=True)

    if run_btn:
        model, label_encoder = load_model()

        if model is None:
            st.error("⚠️ Model files not found. Add `instrument_model.keras` and `label_classes.npy` to the app directory.")
            st.stop()

        # Override settings from sidebar
        CONFIDENCE_THRESHOLD = conf_thresh
        MAX_SILENCE_GAP      = silence_gap
        MIN_PLAY_TIME        = min_play

        with tempfile.TemporaryDirectory() as tmp:
            # Save uploaded file
            audio_path = os.path.join(tmp, uploaded.name)
            with open(audio_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # ── STEP 1: Demucs ──
            st.markdown("---")
            st.markdown("#### 🔄 Processing Pipeline")

            with st.status("Running Demucs source separation…", expanded=True) as status:
                st.write("Splitting song into stems (vocals / drums / bass / other)…")
                stem_folder = run_demucs(audio_path, tmp)
                if stem_folder is None:
                    st.stop()
                status.update(label="✅ Source separation complete", state="complete")

            # ── STEP 2: CNN Scan ──
            st.markdown("**🔬 CNN Scanning stems…**")
            prog  = st.progress(0)
            stxt  = st.empty()

            time_templates = scan_stems(stem_folder, model, label_encoder, prog, stxt)
            stxt.empty()
            prog.progress(1.0)
            st.success(f"✅ Scan complete — {len(time_templates)} raw detection windows found.")

            # ── STEP 3: Merge ──
            results = merge_timestamps(time_templates)

        # ─────────────────────────────────────────
        # RESULTS
        # ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🎼 Detected Instruments")

        if not results:
            st.warning("No instruments detected. Try lowering the Confidence Threshold in the sidebar.")
        else:
            # Summary metrics
            total_blocks   = sum(len(v) for v in results.values())
            longest        = max((e-s for blks in results.values() for s,e in blks), default=0)
            longest_inst   = max(
                results.keys(),
                key=lambda k: max((e-s for s,e in results[k]), default=0)
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Instruments Detected", len(results))
            m2.metric("Total Timestamp Blocks", total_blocks)
            m3.metric("Longest Continuous Block", f"{longest:.1f}s  ({longest_inst.upper()})")

            st.markdown("---")

            # Per-instrument cards
            for inst, blocks in sorted(results.items()):
                code  = inst.lower()
                full  = INSTRUMENT_NAMES.get(code, inst.upper())
                color = INST_COLORS.get(code, "#00C2FF")
                total_dur = sum(e - s for s, e in blocks)

                ts_html = "".join(
                    f"<div class='ts'>▶ {fmt_time(s)} → {fmt_time(e)}"
                    f"<span class='dur' style='margin-left:12px;'>{e-s:.1f}s</span></div>"
                    for s, e in blocks
                )

                st.markdown(
                    f"""<div class='inst-card' style='border-left-color:{color};'>
                        <h4 style='color:{color};'>{inst.upper()} &nbsp;—&nbsp; {full}</h4>
                        <p style='font-size:12px;color:#7A90B0;margin:0 0 6px 0;'>
                            {len(blocks)} block(s) &nbsp;•&nbsp; {total_dur:.1f}s total play time
                        </p>
                        {ts_html}
                    </div>""",
                    unsafe_allow_html=True
                )

            # ── Raw text export ──
            st.markdown("---")
            st.markdown("#### 📥 Export Results")
            raw_lines = [f"=== Instrument Analysis: {uploaded.name} ===\n"]
            for inst, blocks in sorted(results.items()):
                code = inst.lower()
                full = INSTRUMENT_NAMES.get(code, inst.upper())
                raw_lines.append(f"\n{inst.upper()} — {full}")
                for s, e in blocks:
                    raw_lines.append(f"  ▶ [{s:06.1f}s --> {e:06.1f}s]  ({e-s:.1f}s)")
            raw_text = "\n".join(raw_lines)

            st.download_button(
                "⬇️  Download Results (.txt)",
                raw_text,
                file_name=f"{os.path.splitext(uploaded.name)[0]}_instruments.txt",
                mime="text/plain"
            )

else:
    # Placeholder state
    st.markdown("""
    <div style='text-align:center;padding:3rem;color:#7A90B0;'>
        <div style='font-size:3rem;'>🎸</div>
        <p style='font-size:16px;margin-top:0.5rem;'>Upload a WAV, MP3 or FLAC file to get started</p>
        <p style='font-size:13px;'>Supports: vocals · guitar · piano · drums · saxophone · flute · trumpet · violin · cello · clarinet · organ</p>
    </div>
    """, unsafe_allow_html=True)
