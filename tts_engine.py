#!/usr/bin/env python3
"""
TTS Engine — core logic for EPUB → Audiobook conversion.
Extracted from script-from-5060.py for use with the Gradio UI.
"""

import re
import time
import zipfile
import logging
import traceback
import warnings
import subprocess
import soundfile as sf
import numpy as np
import torch
from pathlib import Path
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ─── Constants ─────────────────────────────────────────────────────────────────

CHAR_REPLACEMENTS = [
    ("\u2014", ", "),
    ("\u2013", " to "),
    ("\u2026", "..."),
    ("\u201c", '"'),
    ("\u201d", '"'),
    ("\u2018", "'"),
    ("\u2019", "'"),
]

STRIP_PATTERNS = [
    r"\[TL[^\]]*\]",
    r"\[T/N[^\]]*\]",
    r"\(TL[^\)]*\)",
    r"\[\d+\]",
]

SILENCE_BETWEEN_CHUNKS   = 300   # ms
SILENCE_BETWEEN_CHAPTERS = 1500  # ms


# ─── Text Processing ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    for pattern in STRIP_PATTERNS:
        text = re.sub(pattern, " ", text)
    for old, new in CHAR_REPLACEMENTS:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def get_epub_page_count(epub_path: str) -> int:
    """Return the number of index_split_XXX.html pages in the EPUB."""
    with zipfile.ZipFile(epub_path) as z:
        pages = [f for f in z.namelist() if re.match(r"index_split_\d+\.html", f)]
    return len(pages)


def extract_text_from_epub(
    epub_path: str,
    page_start: int,
    page_end: int,
    epub_prefix: str = "",
) -> list[tuple[str, str]]:
    chapters = []
    with zipfile.ZipFile(epub_path) as z:
        for page_num in range(page_start, page_end + 1):
            fname = f"index_split_{page_num:03d}.html"
            try:
                content = z.read(fname).decode("utf-8", errors="ignore")
            except KeyError:
                continue

            soup = BeautifulSoup(content, "html.parser")
            title_tag = soup.find(["h1", "h2", "h3"])
            title = title_tag.get_text(strip=True) if title_tag else f"Page {page_num}"

            for img in soup.find_all("img"):
                img.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()

            if epub_prefix and text.startswith(epub_prefix):
                text = text[len(epub_prefix):].strip()
            if title and text.startswith(title):
                text = text[len(title):].strip()

            text = clean_text(text)

            if len(text) > 100:
                chapters.append((title, text))
    return chapters


def split_into_chunks(text: str, max_words: int = 200) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, current_count = [], [], 0
    for sentence in sentences:
        wc = len(sentence.split())
        if current_count + wc > max_words and current:
            chunks.append(" ".join(current))
            current, current_count = [sentence], wc
        else:
            current.append(sentence)
            current_count += wc
    if current:
        chunks.append(" ".join(current))
    return chunks


# ─── Audio Processing ──────────────────────────────────────────────────────────

def trim_silence(wav: np.ndarray, sr: int, threshold_db: float = -50.0) -> np.ndarray:
    threshold = 10 ** (threshold_db / 20)
    mask = np.abs(wav) > threshold
    if not mask.any():
        return wav
    start = max(0, np.argmax(mask) - int(sr * 0.01))
    end = min(len(wav), len(mask) - np.argmax(mask[::-1]) + int(sr * 0.01))
    return wav[start:end]


def normalize_volume(wav: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-9:
        return wav
    return wav * (10 ** (target_db / 20) / rms)


def make_silence(sr: int, ms: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000))


# ─── Stitching ─────────────────────────────────────────────────────────────────

def stitch_chapter(chunk_files: list[Path], output_path: Path) -> float:
    combined, sample_rate = [], None
    for f in chunk_files:
        wav, sr = sf.read(f)
        if sample_rate is None:
            sample_rate = sr
        wav = trim_silence(wav, sr)
        wav = normalize_volume(wav)
        combined.append(wav)
        combined.append(make_silence(sr, SILENCE_BETWEEN_CHUNKS))
    if not combined:
        return 0.0
    final = np.concatenate(combined)
    sf.write(str(output_path), final, sample_rate)
    return len(final) / sample_rate


def build_m4b(
    chapter_files: list[tuple[str, Path]],
    output_path: Path,
    output_dir: Path,
    book_title: str = "Audiobook",
    book_artist: str = "Unknown",
    book_album: str = "Audiobook",
) -> bool:
    if not chapter_files:
        return False

    concat_list = output_dir / "concat_list.txt"
    meta_path   = output_dir / "chapters.txt"

    with open(concat_list, "w") as f:
        for _, ch_path in chapter_files:
            f.write(f"file '{ch_path.resolve()}'\n")

    with open(meta_path, "w") as f:
        f.write(";FFMETADATA1\n")
        f.write(f"title={book_title}\n")
        f.write(f"artist={book_artist}\n")
        f.write(f"album={book_album}\n\n")
        offset_ms = 0
        for ch_title, ch_path in chapter_files:
            info = sf.info(str(ch_path))
            duration_ms = int(info.duration * 1000)
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={offset_ms}\n")
            f.write(f"END={offset_ms + duration_ms}\n")
            f.write(f"title={ch_title}\n\n")
            offset_ms += duration_ms + SILENCE_BETWEEN_CHAPTERS

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-i", str(meta_path),
        "-map_metadata", "1",
        "-c:a", "aac", "-b:a", "64k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    concat_list.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return result.returncode == 0


# ─── Main Generation Class ─────────────────────────────────────────────────────

class AudiobookGenerator:
    def __init__(self):
        self.model = None
        self.model_name = None
        self._stop_requested = False

    def load_model(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base") -> str:
        if self.model is not None and self.model_name == model_name:
            return "Model already loaded."
        try:
            from qwen_tts import Qwen3TTSModel
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )
            self.model_name = model_name
            return f"✅ Model loaded: {model_name}"
        except Exception as e:
            return f"❌ Failed to load model: {e}"

    def request_stop(self):
        self._stop_requested = True

    def generate(
        self,
        epub_path: str,
        ref_audio_path: str,
        ref_text: str,
        output_dir: str,
        page_start: int,
        page_end: int,
        epub_prefix: str,
        max_words: int,
        seed: int,
        max_retries: int,
        book_title: str,
        book_artist: str,
        book_album: str,
        language: str,
        progress_callback=None,   # fn(message: str, chunk_done: int, chunk_total: int)
    ) -> dict:
        """
        Run full generation pipeline. Returns dict with keys:
          success, m4b_path, chapter_paths, failed_chunks, message
        """
        self._stop_requested = False

        def log(msg):
            if progress_callback:
                progress_callback(msg, None, None)

        output_dir  = Path(output_dir)
        chunk_dir   = output_dir / "chunks"
        chapter_dir = output_dir / "chapters"
        failure_log = output_dir / "failures.log"
        m4b_path    = output_dir / f"{re.sub(r'[^\\w]', '_', book_title)}.m4b"

        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chapter_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=str(failure_log),
            level=logging.ERROR,
            format="%(asctime)s — %(message)s",
        )

        # ── Extract chapters ──────────────────────────────────────────────────
        log("📖 Loading EPUB...")
        chapters = extract_text_from_epub(epub_path, page_start, page_end, epub_prefix)
        if not chapters:
            return {"success": False, "message": "No chapters found in page range."}
        log(f"   Found {len(chapters)} chapters")

        # ── Split into chunks ─────────────────────────────────────────────────
        log("🧩 Splitting into chunks...")
        all_chunks = []
        chapter_chunk_ranges = {}
        global_idx = 0
        for ch_title, ch_text in chapters:
            chunks = split_into_chunks(ch_text, max_words=max_words)
            start = global_idx
            for i, chunk in enumerate(chunks):
                all_chunks.append((ch_title, i, global_idx, chunk))
                global_idx += 1
            chapter_chunk_ranges[ch_title] = (start, global_idx - 1)
        log(f"   {len(chapters)} chapters → {len(all_chunks)} chunks")

        # ── Load model if needed ──────────────────────────────────────────────
        if self.model is None:
            log("🤖 Loading model...")
            result = self.load_model()
            log(result)
            if "❌" in result:
                return {"success": False, "message": result}

        # ── Generate chunks ───────────────────────────────────────────────────
        log("🎙️  Generating audio chunks...")
        chunk_files = {}
        failed_chunks = []
        chunk_times = []

        existing = sum(1 for _, _, idx, _ in all_chunks if (chunk_dir / f"chunk_{idx:04d}.wav").exists())
        log(f"   {existing} chunks already done, {len(all_chunks) - existing} to generate")

        for done_count, (ch_title, ch_idx, g_idx, text) in enumerate(all_chunks):
            if self._stop_requested:
                log("⛔ Stopped by user.")
                break

            chunk_file = chunk_dir / f"chunk_{g_idx:04d}.wav"
            chunk_files[g_idx] = chunk_file

            if chunk_file.exists():
                if progress_callback:
                    progress_callback(f"⏭️  Skipping chunk {g_idx:04d}", done_count + 1, len(all_chunks))
                continue

            for attempt in range(1, max_retries + 1):
                try:
                    t0 = time.time()
                    torch.manual_seed(seed + attempt - 1)
                    wavs, sr = self.model.generate_voice_clone(
                        text=text,
                        language=language,
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                    )
                    if not wavs:
                        raise ValueError("Model returned empty audio")
                    elapsed = time.time() - t0
                    chunk_times.append(elapsed)
                    sf.write(str(chunk_file), wavs[0], sr)

                    avg = sum(chunk_times[-5:]) / len(chunk_times[-5:])
                    remaining_now = len(all_chunks) - (done_count + 1)
                    eta_min = int(remaining_now * avg / 60)
                    msg = f"✅ {g_idx:04d} | {ch_title} [{ch_idx}] | {len(text.split())} words | {elapsed:.1f}s | ETA ~{eta_min}m"
                    if progress_callback:
                        progress_callback(msg, done_count + 1, len(all_chunks))
                    break
                except Exception as e:
                    if attempt == max_retries:
                        logging.error(f"Chunk {g_idx:04d} failed: {e}\n{traceback.format_exc()}")
                        failed_chunks.append(g_idx)
                        if progress_callback:
                            progress_callback(f"❌ Chunk {g_idx:04d} failed permanently", done_count + 1, len(all_chunks))
                    else:
                        time.sleep(2)

        # ── Stitch chapters ───────────────────────────────────────────────────
        log("📚 Stitching chapters...")
        chapter_flac_files = []
        chapter_paths = []
        for ch_title, _ in chapters:
            start_idx, end_idx = chapter_chunk_ranges[ch_title]
            ch_chunks = [
                chunk_files[i] for i in range(start_idx, end_idx + 1)
                if i in chunk_files and chunk_files[i].exists()
            ]
            if not ch_chunks:
                log(f"   ⚠️  No chunks for {ch_title!r}, skipping")
                continue
            safe = re.sub(r'[^\w\s-]', '', ch_title).strip().replace(' ', '_')
            ch_flac = chapter_dir / f"{safe}.flac"
            duration = stitch_chapter(ch_chunks, ch_flac)
            chapter_flac_files.append((ch_title, ch_flac))
            chapter_paths.append(str(ch_flac))
            log(f"   ✅ {ch_title} → {ch_flac.name} ({duration:.1f}s)")

        # ── Build M4B ─────────────────────────────────────────────────────────
        log("🎬 Building M4B...")
        success = build_m4b(
            chapter_flac_files, m4b_path, output_dir,
            book_title=book_title, book_artist=book_artist, book_album=book_album,
        )

        if success:
            log(f"✅ Done! M4B saved to: {m4b_path}")
        else:
            log("❌ M4B build failed.")

        return {
            "success": success,
            "m4b_path": str(m4b_path) if success else None,
            "chapter_paths": chapter_paths,
            "failed_chunks": failed_chunks,
            "message": f"Done. {len(failed_chunks)} chunks failed." if failed_chunks else "Done!",
        }
