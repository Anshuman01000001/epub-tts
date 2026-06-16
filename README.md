# 📚 Audiobook TTS

Convert EPUB files into full audiobooks using AI voice cloning.  
Generates per-chapter FLAC files and a combined M4B with chapter markers.

Powered by [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) — runs fully locally, no API keys or cloud services required.

---

## Features

- Voice cloning from a short reference audio clip
- Resumable generation — safe to stop and restart at any time
- Per-chapter FLAC output + combined M4B with chapter markers
- Clean Gradio UI or headless script mode
- Automatic translator note removal and Unicode cleaning
- Volume normalization and silence trimming between chunks

---

## Requirements

- Linux (tested on Arch)
- Python 3.12
- NVIDIA GPU with 6GB+ VRAM (tested on RTX 5060 8GB and RTX 5070 Ti 16GB)
- CUDA 12.x or 13.x
- ffmpeg
- sox

---

## Installation

**1. Clone the repo**
```bash
git clone https://github.com/Anshuman01000001/audiobook-tts.git
cd audiobook-tts
```

**2. Create a virtual environment**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**3. Install PyTorch (cu128 build)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**4. Install dependencies**
```bash
pip install qwen-tts ebooklib beautifulsoup4 soundfile numpy tqdm gradio
```

**5. Install system packages**
```bash
# Arch
sudo pacman -S ffmpeg sox

# Ubuntu/Debian
sudo apt install ffmpeg sox
```

**6. Download the model**
```bash
pip install huggingface_hub
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

---

## Usage

### Gradio UI (recommended)

```bash
source .venv/bin/activate
python app.py
```

Opens at `http://localhost:7860`. Upload your EPUB, reference audio, set the page range and output directory, then click **Generate**.

### Headless script

Edit the config section at the top of `script.py`:

```python
EPUB_PATH      = "/path/to/your/book.epub"
REF_AUDIO_PATH = "/path/to/reference.wav"
OUTPUT_DIR     = Path("/path/to/output")
PAGE_START     = 5     # page = chapter + 4 for most EPUBs
PAGE_END       = 217
```

Then run:
```bash
python script.py
```

---

## Reference Audio

For best results, use a clean 10-30 second WAV clip with:
- No background music or noise
- Clear, natural speech
- The same accent/tone you want in the output

Set `REF_TEXT` to an exact transcript of what the clip says.

You can separate vocals from a mixed audio file using [Demucs](https://github.com/facebookresearch/demucs):
```bash
pip install demucs
demucs your_audio.mp3
```

---

## EPUB Structure

Most EPUBs from fan translation sites use `index_split_XXX.html` page numbering.  
For many novels: `page = chapter + 4` (Chapter 1 = page 5, Chapter 2 = page 6, etc.)

Check your EPUB's structure by uploading it in the UI — it will auto-detect the page count and preview chapter titles.

---

## Output

```
output/
├── chunks/          ← per-chunk WAVs (resumable, deleted after successful M4B)
├── chapters/        ← per-chapter FLACs
├── YourBook.m4b     ← final audiobook with chapter markers
└── failures.log     ← any chunks that failed after all retries
```

The M4B file works natively in:
- Apple Podcasts / Books
- VLC
- Most podcast apps (Pocket Casts, Overcast, etc.)

To convert to MP4 for YouTube:
```bash
ffmpeg -loop 1 -i cover.jpg -i YourBook.m4b \
  -c:v libx264 -tune stillimage -crf 51 -preset ultrafast \
  -c:a copy -shortest output.mp4
```

---

## Performance

| GPU | Chunk time | 200-chapter book |
|-----|-----------|-----------------|
| RTX 5060 8GB | ~60s/chunk | ~20h |
| RTX 5070 Ti 16GB | ~25s/chunk | ~8h |

Generation is resumable — you can stop and restart without redoing completed chunks.

---

## Known Issues

- **flash-attn** does not install on CUDA 13.x — falls back to SDPA automatically
- **onnxruntime-gpu** has no official wheel for CUDA 13.x — speaker encoder runs on CPU
- GPU utilization is ~50% due to the above CPU bottleneck

---

## License

This project is released under the MIT License.  
The Qwen3-TTS model is subject to its own license — see the [model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).

Ensure you have the rights to any EPUB you convert. This tool is intended for personal use with content you own or have permission to use.
