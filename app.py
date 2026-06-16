#!/usr/bin/env python3
"""
Audiobook TTS — Gradio UI
Run with: python app.py
"""

import gradio as gr
import zipfile
import re
import threading
from pathlib import Path
from tts_engine import AudiobookGenerator, get_epub_page_count

# ─── Global state ──────────────────────────────────────────────────────────────

generator = AudiobookGenerator()
generation_thread = None
log_lines = []
progress_done = 0
progress_total = 0

# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_epub_info(epub_file):
    """Return page count and first few chapter titles for preview."""
    if epub_file is None:
        return "Upload an EPUB to see info.", 5, 10
    try:
        count = get_epub_page_count(epub_file.name)
        titles = []
        with zipfile.ZipFile(epub_file.name) as z:
            from bs4 import BeautifulSoup
            for i in range(1, min(8, count)):
                fname = f"index_split_{i:03d}.html"
                try:
                    content = z.read(fname).decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(content, "html.parser")
                    tag = soup.find(["h1", "h2", "h3"])
                    if tag:
                        titles.append(f"  page {i}: {tag.get_text(strip=True)}")
                except KeyError:
                    pass
        preview = f"📚 {count} pages found\n\nFirst chapters:\n" + "\n".join(titles)
        return preview, 5, count
    except Exception as e:
        return f"Error reading EPUB: {e}", 5, 10


def progress_callback(message, done, total):
    global log_lines, progress_done, progress_total
    log_lines.append(message)
    if len(log_lines) > 200:
        log_lines = log_lines[-200:]
    if done is not None:
        progress_done = done
    if total is not None:
        progress_total = total


def get_log():
    return "\n".join(log_lines[-60:])


def get_progress():
    if progress_total == 0:
        return 0
    return progress_done / progress_total


def load_model(model_name):
    global log_lines
    log_lines.append(f"Loading model: {model_name}...")
    result = generator.load_model(model_name)
    log_lines.append(result)
    return get_log()


def start_generation(
    epub_file,
    ref_audio,
    ref_text,
    output_dir,
    page_start,
    page_end,
    epub_prefix,
    max_words,
    seed,
    max_retries,
    book_title,
    book_artist,
    book_album,
    language,
):
    global generation_thread, log_lines, progress_done, progress_total

    if epub_file is None:
        return "❌ Please upload an EPUB file.", get_log()
    if ref_audio is None:
        return "❌ Please upload a reference audio file.", get_log()
    if not output_dir.strip():
        return "❌ Please set an output directory.", get_log()
    if generation_thread and generation_thread.is_alive():
        return "⚠️ Generation already running.", get_log()

    log_lines = []
    progress_done = 0
    progress_total = 0

    def run():
        generator.generate(
            epub_path=epub_file.name,
            ref_audio_path=ref_audio.name,
            ref_text=ref_text,
            output_dir=output_dir,
            page_start=int(page_start),
            page_end=int(page_end),
            epub_prefix=epub_prefix,
            max_words=int(max_words),
            seed=int(seed),
            max_retries=int(max_retries),
            book_title=book_title,
            book_artist=book_artist,
            book_album=book_album,
            language=language,
            progress_callback=progress_callback,
        )

    generation_thread = threading.Thread(target=run, daemon=True)
    generation_thread.start()
    return "🎙️ Generation started!", get_log()


def stop_generation():
    generator.request_stop()
    return "⛔ Stop requested — will halt after current chunk."


# ─── UI ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&family=JetBrains+Mono:wght@400&display=swap');

:root {
    --bg:        #0f0e0c;
    --surface:   #1a1814;
    --border:    #2e2a22;
    --gold:      #c9a84c;
    --gold-dim:  #7a6330;
    --text:      #e8e0d0;
    --muted:     #7a7060;
    --danger:    #8b3a3a;
    --success:   #3a6b4a;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Crimson Pro', Georgia, serif !important;
}

h1.title {
    font-family: 'Cinzel', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--gold);
    letter-spacing: 0.08em;
    text-align: center;
    margin: 0.5rem 0 0.2rem;
    text-shadow: 0 0 40px rgba(201,168,76,0.3);
}

p.subtitle {
    font-family: 'Crimson Pro', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--muted);
    text-align: center;
    margin: 0 0 1.5rem;
    letter-spacing: 0.03em;
}

.panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 1.2rem !important;
}

label {
    font-family: 'Cinzel', serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    color: var(--gold-dim) !important;
    text-transform: uppercase !important;
}

input, textarea, select {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Crimson Pro', serif !important;
    font-size: 1rem !important;
    border-radius: 2px !important;
}

input:focus, textarea:focus {
    border-color: var(--gold-dim) !important;
    outline: none !important;
    box-shadow: 0 0 0 1px var(--gold-dim) !important;
}

button.primary {
    background: var(--gold) !important;
    color: #0f0e0c !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.7rem 1.5rem !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}

button.primary:hover {
    background: #e0bc60 !important;
}

button.secondary {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.2rem !important;
    cursor: pointer !important;
    transition: border-color 0.2s, color 0.2s !important;
}

button.secondary:hover {
    border-color: var(--muted) !important;
    color: var(--text) !important;
}

button.stop {
    background: var(--danger) !important;
    color: #f0d0d0 !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.2rem !important;
}

.log-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #080807 !important;
    color: #9a9080 !important;
    border-color: var(--border) !important;
    line-height: 1.6 !important;
}

.progress-bar {
    background: var(--border) !important;
    border-radius: 1px !important;
}

.progress-bar > div {
    background: var(--gold) !important;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
}

.status-ok  { color: #6db080; }
.status-err { color: #b06060; }

/* Gradio overrides */
.gr-box, .gr-form { background: transparent !important; border: none !important; }
.gr-padded { padding: 0 !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Audiobook TTS") as app:

    gr.HTML("""
        <h1 class='title'>⚗ Audiobook TTS</h1>
        <p class='subtitle'>EPUB → M4B with voice cloning via Qwen3-TTS</p>
    """)

    with gr.Row():

        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="panel"):

            gr.Markdown("### Files")

            epub_file = gr.File(
                label="EPUB File",
                file_types=[".epub"],
            )
            epub_info = gr.Textbox(
                label="EPUB Info",
                interactive=False,
                lines=5,
                value="Upload an EPUB to see chapter info.",
            )
            ref_audio = gr.File(
                label="Reference Audio (WAV)",
                file_types=[".wav", ".mp3", ".flac"],
            )
            ref_text = gr.Textbox(
                label="Reference Text (what the audio says)",
                lines=4,
                value=(
                    "Whichever potion I choose, the only thing bound to get me home is knowledge of mysticism. "
                    "That rules out Sleepless and Corpse Collector. It's between Mystery Prior and Seer. "
                    "But which would be more useful? Man this is even more stressful than choosing a major in college."
                ),
            )

            gr.HTML("<hr class='divider'>")
            gr.Markdown("### Output")

            output_dir = gr.Textbox(
                label="Output Directory",
                value="/home/anshuman/Desktop/LOTM/audio",
            )
            book_title  = gr.Textbox(label="Book Title",  value="Lord of Mysteries Vol.1: Clown")
            book_artist = gr.Textbox(label="Author",       value="Cuttlefish That Loves Diving")
            book_album  = gr.Textbox(label="Series/Album", value="Lord of Mysteries")

        # ── Right column: settings + controls ────────────────────────────────
        with gr.Column(scale=1, elem_classes="panel"):

            gr.Markdown("### Settings")

            with gr.Row():
                page_start = gr.Number(label="Page Start", value=5,  precision=0)
                page_end   = gr.Number(label="Page End",   value=10, precision=0)

            epub_prefix = gr.Textbox(
                label="EPUB Prefix to Strip",
                value="Lord of Mysteries Volume 1: Clown",
            )
            language = gr.Dropdown(
                label="Language",
                choices=["English", "Chinese", "Japanese", "Korean", "German",
                         "French", "Russian", "Portuguese", "Spanish", "Italian"],
                value="English",
            )

            with gr.Row():
                max_words   = gr.Number(label="Max Words / Chunk", value=200, precision=0)
                max_retries = gr.Number(label="Max Retries",        value=3,   precision=0)
                seed        = gr.Number(label="Seed", value=559643366572128, precision=0)

            gr.HTML("<hr class='divider'>")
            gr.Markdown("### Model")

            model_name = gr.Dropdown(
                label="Model",
                choices=[
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                ],
                value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            )

            with gr.Row():
                load_btn  = gr.Button("Load Model",  variant="secondary")
                start_btn = gr.Button("▶ Generate",  variant="primary")
                stop_btn  = gr.Button("■ Stop",       variant="stop")

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
            )

            gr.HTML("<hr class='divider'>")
            gr.Markdown("### Progress")

            progress_bar = gr.Slider(
                label="Chunks complete",
                minimum=0, maximum=1, value=0,
                interactive=False,
            )

            log_box = gr.Textbox(
                label="Log",
                lines=18,
                interactive=False,
                elem_classes="log-box",
            )

            refresh_btn = gr.Button("↻ Refresh Log", variant="secondary")

    # ─── Events ──────────────────────────────────────────────────────────────

    epub_file.change(
        fn=get_epub_info,
        inputs=[epub_file],
        outputs=[epub_info, page_start, page_end],
    )

    load_btn.click(
        fn=load_model,
        inputs=[model_name],
        outputs=[log_box],
    )

    start_btn.click(
        fn=start_generation,
        inputs=[
            epub_file, ref_audio, ref_text, output_dir,
            page_start, page_end, epub_prefix,
            max_words, seed, max_retries,
            book_title, book_artist, book_album, language,
        ],
        outputs=[status_box, log_box],
    )

    stop_btn.click(
        fn=stop_generation,
        outputs=[status_box],
    )

    refresh_btn.click(
        fn=lambda: (get_log(), get_progress()),
        outputs=[log_box, progress_bar],
    )

    # Auto-refresh every 5s while app is open - updated
    gr.Timer(5).tick(
        fn=lambda: (get_log(), get_progress()),
        outputs=[log_box, progress_bar],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
