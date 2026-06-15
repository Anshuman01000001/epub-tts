# LOTM Audiobook TTS

Convert *Lord of the Mysteries* (or any EPUB) into a fully chaptered M4B audiobook using [Qwen3-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) voice cloning — no ComfyUI required.

Built and optimized for an **RTX 5070 Ti + AMD 9800X3D** but runs on any CUDA-capable GPU with 8 GB+ VRAM.

---

## Features

- 🎙️ **Voice cloning** — provide a reference WAV and the narrator matches that voice throughout
- 📖 **EPUB-native** — extracts chapters directly from the EPUB file structure
- 💾 **Resumable** — already-generated chunks are skipped on restart; safe to cancel at any time
- 🏎️ **Optimized** — Flash Attention 2, `torch.compile`, async disk I/O, parallel chunk reads
- 📚 **M4B output** — single audiobook file with chapter markers, title, and artist metadata
- 🧹 **Clean text** — strips translator notes, footnotes, and fixes Unicode punctuation before synthesis

---

## Requirements

- Python 3.10+
- CUDA 12.8 (for RTX 4000/5000 series; adjust for your GPU)
- `ffmpeg` installed and on PATH
- A reference voice WAV file (~10–30 seconds of clean speech)

---

## Setup

```bash
# 1. Clone this repo
git clone https://github.com/yourusername/lotm-tts.git
cd lotm-tts

# 2. Clone the Qwen-TTS backend into the same folder
git clone https://github.com/flybirdxx/ComfyUI-Qwen-TTS .

# 3. Create a self-contained virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 4. Install PyTorch with CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5. Install Flash Attention 2 (takes 5–15 min to compile)
pip install flash-attn --no-build-isolation

# 6. Install remaining dependencies
pip install transformers==4.57.3 soundfile ebooklib beautifulsoup4 \
            lxml huggingface_hub accelerate tqdm
```

> **Note:** `transformers==4.57.3` is pinned — version 5+ breaks Qwen3-TTS.

---

## Model Download

Models are downloaded once and cached by HuggingFace (~14 GB for 7.1B):

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-7.1B-Base
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz
```

For lower VRAM (< 12 GB), use the 1.7B model instead:

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

And update `MODEL_CHOICE` in the script accordingly.

---

## Configuration

Edit the config block at the top of `lotm_tts_fast.py`:

```python
EPUB_PATH      = "/path/to/your/book.epub"
REF_AUDIO_PATH = "/path/to/reference_voice.wav"
REF_TEXT       = "Transcript of the reference audio clip."

PAGE_START     = 40    # First HTML page index in the EPUB to convert
PAGE_END       = 50    # Last HTML page index (inclusive)
MAX_WORDS      = 400   # Words per TTS chunk — increase for fewer model calls
```

To find your `PAGE_START` / `PAGE_END` values, unzip the EPUB and look at the `index_split_NNN.html` filenames.

---

## Usage

```bash
source .venv/bin/activate
python lotm_tts_fast.py
```

Output structure:

```
audio/
├── chunks/          # Per-chunk WAV files (intermediate, resumable)
├── chapters/        # Per-chapter FLAC files
├── LOTM_Vol1_Klein.m4b   # Final audiobook
└── failures.log     # Any chunks that failed after all retries
```

---

## Performance

Tested on RTX 5070 Ti (16 GB VRAM) + AMD Ryzen 9 9800X3D:

| Setting | Time per chunk | ~1400 chapters |
|---|---|---|
| 7.1B + FA2 + compile, 400 words | ~25–40s | 3–5 days |
| 1.7B + FA2 + compile, 400 words | ~10–18s | 1–2 days |

> The first chapter takes ~60s longer while `torch.compile` traces the model — every chapter after is faster.

---

## Project Structure

```
lotm-tts/
├── .venv/               # Self-contained Python environment
├── qwen_tts/            # Qwen3-TTS backend (cloned from ComfyUI-Qwen-TTS)
├── lotm_tts_fast.py     # Main conversion script
├── audio/               # Generated output
│   ├── chunks/
│   ├── chapters/
│   └── LOTM_Vol1_Klein.m4b
└── README.md
```

---

## Troubleshooting

**`ImportError: cannot import name 'Qwen3TTSModel'`**
Make sure you cloned the ComfyUI-Qwen-TTS repo *into* the same folder as the script so the `qwen_tts/` package is on the Python path.

**Flash Attention 2 not loading**
The script falls back to SDPA automatically. To manually install FA2:
```bash
pip install flash-attn --no-build-isolation
```
Requires CUDA toolkit headers — install via `sudo apt install cuda-toolkit-12-8`.

**`transformers` version conflict**
This project requires exactly `transformers==4.57.3`. If something else upgraded it, run:
```bash
pip install transformers==4.57.3 --force-reinstall
```

**Chunks failing repeatedly**
Check `audio/failures.log` for the full traceback. Common cause is a chunk with unusual Unicode characters — the script's `CHAR_REPLACEMENTS` list can be extended to handle them.

---

## Credits

- [Qwen3-TTS](https://huggingface.co/Qwen) by Alibaba DAMO Academy
- [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) by flybirdxx
- *Lord of the Mysteries* by Cuttlefish That Loves Diving
