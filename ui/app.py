"""
RVCEdge — Gradio WebUI
Real-time RVC voice changer with ContentVec + RMVPE + WASAPI.
"""

import sys
import os
import time
import threading
import tempfile
import gradio as gr
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import RVCEdgeEngine, DEVICE_SR
from core.audio_io import AudioStream, list_devices


# ── Global state ──────────────────────────────────────────────────────────────

engine: RVCEdgeEngine = None
stream: AudioStream = None
stream_lock = threading.Lock()
MODELS_DIR = Path(__file__).parent.parent / "models"
BASE_MODELS_DIR = Path(__file__).parent.parent / "base_models"


def get_engine() -> RVCEdgeEngine:
    global engine
    if engine is None:
        engine = RVCEdgeEngine(device="auto")
    return engine


# ── Helpers ────────────────────────────────────────────────────────────────────

def scan_models():
    if not MODELS_DIR.exists():
        return []
    files = sorted(list(MODELS_DIR.glob("*.pth")) + list(MODELS_DIR.glob("*.safetensors")))
    return [str(p) for p in files]


def scan_indexes():
    idxs = sorted(MODELS_DIR.glob("*.index")) if MODELS_DIR.exists() else []
    return ["None"] + [str(p) for p in idxs]


def get_device_choices():
    try:
        inputs, outputs = list_devices()
        return ([f"{i}: {name}" for i, name in inputs],
                [f"{i}: {name}" for i, name in outputs])
    except Exception as e:
        return [f"Error: {e}"], [f"Error: {e}"]


def parse_device_id(s: str) -> int:
    return int(s.split(":")[0])


# ── Tab 1: Real-time ───────────────────────────────────────────────────────────

def load_model(pth_path: str, index_path: str, pitch: float,
               index_rate: float, protect: float, sid: int) -> str:
    if not pth_path or not Path(pth_path).exists():
        return "Select a valid .pth model file."
    try:
        eng = get_engine()
        eng.pitch_shift = pitch
        eng.index_rate = index_rate
        eng.protect = protect
        eng.sid = sid
        idx = None if index_path == "None" else index_path
        eng.load_model(pth_path, index_path=idx)
        return (f"Model loaded: {Path(pth_path).name}\n"
                f"Version: {eng.model_version} | SR: {eng.model_sr}Hz | "
                f"Pitch: {pitch:+.0f}st | Device: {eng.device_str.upper()}")
    except Exception as e:
        return f"Error: {e}"


def update_pitch(pitch: float) -> str:
    eng = get_engine()
    eng.pitch_shift = pitch
    return f"Pitch shift updated: {pitch:+.1f} semitones"


def start_stream(in_dev: str, out_dev: str, wasapi_excl: bool) -> str:
    global stream
    with stream_lock:
        if stream is not None:
            return "Already running."
        eng = get_engine()
        if not eng.is_ready:
            return "Load a model first."
        eng.warmup(3)
        try:
            stream = AudioStream(
                input_device=parse_device_id(in_dev),
                output_device=parse_device_id(out_dev),
                process_fn=eng.process_chunk,
                use_wasapi_exclusive=wasapi_excl,
            )
            stream.start()
            return f"Running — {in_dev.split(':', 1)[1].strip()}"
        except Exception as e:
            stream = None
            return f"Failed: {e}"


def stop_stream() -> str:
    global stream
    with stream_lock:
        if stream is None:
            return "Not running."
        try:
            stream.stop()
        except Exception:
            pass
        stream = None
        return "Stopped."


def get_status() -> str:
    if stream is None or not stream.running:
        return "Stopped"
    proc = stream.processing_time_ms
    lat = stream.total_latency_ms
    under = stream.output_underruns
    return (f"Running\n"
            f"Processing: {proc:.1f}ms\n"
            f"Est. latency: {lat:.0f}ms\n"
            f"Underruns: {under}")


# ── Tab 2: File conversion ─────────────────────────────────────────────────────

def convert_file(source, reference_model, reference_index, pitch,
                 index_rate, protect, sid):
    import torchaudio.functional as taF
    import torch

    if source is None:
        return None, "Upload source audio."
    if not reference_model or not Path(reference_model).exists():
        return None, "Select a .pth model."
    try:
        eng = get_engine()
        eng.pitch_shift = pitch
        eng.index_rate = index_rate
        eng.protect = protect
        eng.sid = sid
        idx = None if reference_index == "None" else reference_index
        eng.load_model(reference_model, index_path=idx)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        eng.convert_file(source, out_path, chunk_s=5.0)
        return out_path, f"Done. Output: {out_path}"
    except Exception as e:
        return None, f"Error: {e}"


# ── Build UI ───────────────────────────────────────────────────────────────────

def build_ui():
    in_choices, out_choices = get_device_choices()
    models = scan_models()
    indexes = scan_indexes()

    import torch
    gpu_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "CPU")

    with gr.Blocks(title="RVCEdge") as demo:
        gr.Markdown(
            f"# RVCEdge\n"
            f"**Real-time RVC** · ContentVec (no fairseq) · RMVPE · WASAPI  \n"
            f"GPU: **{gpu_name}**"
        )

        with gr.Tabs():

            # ── Tab 1: Real-time ──────────────────────────────────────────────
            with gr.Tab("Real-time"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Model")
                        pth_dd = gr.Dropdown(
                            choices=models,
                            value=models[0] if models else None,
                            label="Voice model (.pth)",
                            allow_custom_value=True,
                        )
                        idx_dd = gr.Dropdown(
                            choices=indexes,
                            value="None",
                            label="FAISS index (.index) — optional, improves similarity",
                        )
                        with gr.Row():
                            refresh_btn = gr.Button("Refresh models", size="sm")

                        gr.Markdown("### Voice settings")
                        pitch_sl = gr.Slider(-24, 24, value=12, step=0.5,
                                             label="Pitch shift (semitones) — male→female: +10 to +12")
                        index_rate_sl = gr.Slider(0, 1, value=0.75, step=0.05,
                                                   label="Index rate (0=off, 1=full similarity)")
                        protect_sl = gr.Slider(0, 0.5, value=0.33, step=0.01,
                                               label="Protect consonants (0.33 recommended)")
                        sid_nb = gr.Number(value=0, label="Speaker ID (usually 0)", precision=0)

                        load_btn = gr.Button("Load model", variant="primary")
                        load_status = gr.Textbox(label="Status", interactive=False, lines=3)

                        gr.Markdown("### Audio devices")
                        in_dev = gr.Dropdown(in_choices,
                                             value=in_choices[0] if in_choices else None,
                                             label="Input (microphone)")
                        out_dev = gr.Dropdown(out_choices,
                                              value=out_choices[0] if out_choices else None,
                                              label="Output (speakers)")
                        wasapi_cb = gr.Checkbox(value=True,
                                                label="WASAPI Exclusive (lower latency)")

                        with gr.Row():
                            start_btn = gr.Button("▶ Start", variant="primary")
                            stop_btn = gr.Button("■ Stop", variant="stop")

                    with gr.Column(scale=1):
                        gr.Markdown("### Monitor")
                        status_box = gr.Textbox(label="Stream status", lines=6,
                                                interactive=False)
                        status_btn = gr.Button("Refresh status")
                        pitch_upd_btn = gr.Button("Apply pitch change live")
                        pitch_status = gr.Textbox(label="", interactive=False, lines=1)

                def do_refresh():
                    models = scan_models()
                    indexes = scan_indexes()
                    return gr.Dropdown(choices=models), gr.Dropdown(choices=indexes)

                refresh_btn.click(do_refresh, outputs=[pth_dd, idx_dd])
                load_btn.click(load_model,
                               inputs=[pth_dd, idx_dd, pitch_sl, index_rate_sl, protect_sl, sid_nb],
                               outputs=load_status)
                start_btn.click(start_stream,
                                inputs=[in_dev, out_dev, wasapi_cb],
                                outputs=load_status)
                stop_btn.click(stop_stream, outputs=load_status)
                status_btn.click(get_status, outputs=status_box)
                pitch_upd_btn.click(update_pitch, inputs=[pitch_sl], outputs=pitch_status)

            # ── Tab 2: File conversion ────────────────────────────────────────
            with gr.Tab("File conversion"):
                with gr.Row():
                    with gr.Column():
                        src_audio = gr.Audio(label="Source audio", type="filepath")
                        model_dd_f = gr.Dropdown(
                            choices=models, value=models[0] if models else None,
                            label="Voice model (.pth)", allow_custom_value=True
                        )
                        idx_dd_f = gr.Dropdown(
                            choices=indexes, value="None", label="FAISS index (optional)"
                        )
                        pitch_f = gr.Slider(-24, 24, value=12, step=0.5,
                                            label="Pitch shift (semitones)")
                        idx_rate_f = gr.Slider(0, 1, value=0.75, step=0.05,
                                               label="Index rate")
                        protect_f = gr.Slider(0, 0.5, value=0.33, step=0.01,
                                              label="Protect")
                        sid_f = gr.Number(value=0, label="Speaker ID", precision=0)
                        conv_btn = gr.Button("Convert", variant="primary")
                    with gr.Column():
                        out_audio = gr.Audio(label="Output")
                        conv_info = gr.Textbox(label="Info", interactive=False)

                conv_btn.click(
                    convert_file,
                    inputs=[src_audio, model_dd_f, idx_dd_f, pitch_f,
                            idx_rate_f, protect_f, sid_f],
                    outputs=[out_audio, conv_info],
                )

            # ── Tab 3: Setup guide ────────────────────────────────────────────
            with gr.Tab("Setup"):
                gr.Markdown(f"""
### Setup guide

**1. Download base models** (one-time, ~500MB total):
```bash
python download_base.py
```
Downloads:
- `base_models/rmvpe.pt` — pitch extractor (~100MB)
- `base_models/contentvec/` — ContentVec feature extractor (~360MB, auto via transformers)

**2. Place your RVC models:**
Copy your `.pth` files (and optionally `.index` files) into the `models/` folder.

**3. Recommended settings for male→female:**
- Pitch shift: **+10 to +12 semitones**
- Index rate: **0.75** (if you have a .index file)
- Protect: **0.33**
- WASAPI Exclusive: enabled (if available)

**4. Expected latency (RTX 4070):**
- Feature extraction (ContentVec): ~30ms
- RMVPE pitch: ~15ms
- VITS inference: ~20ms
- Audio buffer: ~200ms
- **Total: ~265ms**

### What's different from Vonovox / Applio

| Feature | RVCEdge | Vonovox | Applio |
|---------|---------|---------|--------|
| fairseq dependency | ❌ No | ✅ Yes | ✅ Yes |
| ContentVec (transformers) | ✅ | ❌ | ✅ v3.9+ |
| Gradio 6.x WebUI | ✅ | ❌ | ✅ |
| WASAPI Exclusive | ✅ | ✅ | ❌ |
| Live pitch adjust | ✅ | ❌ | ❌ |
| FAISS index | ✅ | ✅ | ✅ |

### Model sources
- [AI Hub](https://aihub.gg) — thousands of RVC v2 models
- [voice-models.com](https://voice-models.com) — community models
""")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        theme=gr.themes.Soft(),
    )
