#!/usr/bin/env python3
"""
LOTM Audiobook TTS Script
Converts an EPUB to audio using Qwen3-TTS voice cloning.
Outputs per-chapter FLACs + a combined M4B with chapter markers.
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
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────────

EPUB_PATH       = "/home/anshuman/Documents/local-ai-projects/audiobook-lotm/epubs/Clown-LotM-Vol.1.epub"
REF_AUDIO_PATH  = "/home/anshuman/Documents/local-ai-projects/audiobook-lotm/ref_voice/vocals.wav"
REF_TEXT        = (
    "Whichever potion I choose, the only thing bound to get me home is knowledge of mysticism. "
    "That rules out Sleepless and Corpse Collector. It's between Mystery Prior and Seer. "
    "But which would be more useful? Man this is even more stressful than choosing a major in college."
)
OUTPUT_DIR      = Path("/home/anshuman/Documents/local-ai-projects/audiobook-lotm/audio/")
CHUNK_DIR       = OUTPUT_DIR / "chunks"
CHAPTER_DIR     = OUTPUT_DIR / "chapters"
FINAL_OUTPUT    = OUTPUT_DIR / "LOTM_Vol1_Klein.m4b"
FAILURE_LOG     = OUTPUT_DIR / "failures.log"

SEED            = 559643366572128
MODEL_CHOICE    = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAX_WORDS       = 200             # words per TTS chunk
MAX_RETRIES     = 3               # retry failed chunks this many times
PAGE_START      = 5              # index_split_005.html  (Chapter 5 start) - 5 would be chapter 1
PAGE_END        = 217              # index_split_011.html  (Chapter 213 end, inclusive)
LANGUAGE        = "English"

# Silence durations (ms)
SILENCE_BETWEEN_CHUNKS   = 300
SILENCE_BETWEEN_CHAPTERS = 1500

# Prefix stripped from every HTML file
EPUB_PREFIX = "Lord of Mysteries Volume 1: Clown"

# Special character replacements before TTS
CHAR_REPLACEMENTS = [
    ("\u2014", ", "),    # em dash → comma space
    ("\u2013", " to "),  # en dash → " to "
    ("\u2026", "..."),   # ellipsis character → three dots
    ("\u201c", '"'),     # left double quote
    ("\u201d", '"'),     # right double quote
    ("\u2018", "'"),     # left single quote
    ("\u2019", "'"),     # right single quote
]

# Regex patterns to strip entirely
STRIP_PATTERNS = [
    r"\[TL[^\]]*\]",    # translator notes [TL: ...]
    r"\[T/N[^\]]*\]",   # translator notes [T/N: ...]
    r"\(TL[^\)]*\)",    # translator notes (TL: ...)
    r"\[\d+\]",         # footnote markers [1], [2]
]

# ─── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename=str(OUTPUT_DIR / "failures.log"),
    level=logging.ERROR,
    format="%(asctime)s — %(message)s",
)

# ─── Text Processing ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip translator notes, fix special characters for TTS."""
    for pattern in STRIP_PATTERNS:
        text = re.sub(pattern, " ", text)
    for old, new in CHAR_REPLACEMENTS:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_epub(epub_path: str, page_start: int, page_end: int) -> list[tuple[str, str]]:
    """Returns list of (chapter_title, chapter_text) for the given page range (inclusive)."""
    chapters = []
    with zipfile.ZipFile(epub_path) as z:
        for page_num in range(page_start, page_end + 1):
            fname = f"index_split_{page_num:03d}.html"
            try:
                content = z.read(fname).decode("utf-8", errors="ignore")
            except KeyError:
                print(f"   ⚠️  {fname} not found, skipping")
                continue

            soup = BeautifulSoup(content, "html.parser")
            title_tag = soup.find(["h1", "h2", "h3"])
            title = title_tag.get_text(strip=True) if title_tag else f"Page {page_num}"

            for img in soup.find_all("img"):
                img.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()

            if text.startswith(EPUB_PREFIX):
                text = text[len(EPUB_PREFIX):].strip()
            if title and text.startswith(title):
                text = text[len(title):].strip()

            text = clean_text(text)

            if len(text) > 100:
                chapters.append((title, text))
    return chapters


def split_into_chunks(text: str, max_words: int = 200) -> list[str]:
    """Split text into chunks at sentence boundaries."""
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
    """Trim leading and trailing silence."""
    threshold = 10 ** (threshold_db / 20)
    mask = np.abs(wav) > threshold
    if not mask.any():
        return wav
    start = max(0, np.argmax(mask) - int(sr * 0.01))
    end = min(len(wav), len(mask) - np.argmax(mask[::-1]) + int(sr * 0.01))
    return wav[start:end]


def normalize_volume(wav: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize to a target RMS level."""
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-9:
        return wav
    return wav * (10 ** (target_db / 20) / rms)


def make_silence(sr: int, ms: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000))


# ─── Generation ────────────────────────────────────────────────────────────────

def generate_chunk(model, text: str, seed: int):
    torch.manual_seed(seed)
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=LANGUAGE,
        ref_audio=REF_AUDIO_PATH,
        ref_text=REF_TEXT,
    )
    if not wavs:
        raise ValueError("Model returned empty audio")
    return wavs[0], sr


def generate_with_retry(model, text: str, idx: int) -> tuple[np.ndarray, int]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return generate_chunk(model, text, SEED + attempt - 1)
        except Exception as e:
            tqdm.write(f"   ⚠️  Chunk {idx:04d} attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                logging.error(f"Chunk {idx:04d} failed all {MAX_RETRIES} attempts: {e}\n{traceback.format_exc()}")
                raise
            time.sleep(2)


# ─── Stitching ─────────────────────────────────────────────────────────────────

def stitch_chapter(chunk_files: list[Path], output_path: Path) -> float:
    """Stitch chunks into a per-chapter FLAC. Returns duration in seconds."""
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


def build_m4b(chapter_files: list[tuple[str, Path]], output_path: Path):
    """Combine per-chapter FLACs into a single M4B with chapter markers."""
    if not chapter_files:
        print("❌ No chapter files to combine")
        return

    concat_list = OUTPUT_DIR / "concat_list.txt"
    meta_path   = OUTPUT_DIR / "chapters.txt"

    with open(concat_list, "w") as f:
        for _, ch_path in chapter_files:
            f.write(f"file '{ch_path.resolve()}'\n")

    with open(meta_path, "w") as f:
        f.write(";FFMETADATA1\n")
        f.write("title=Lord of Mysteries Vol.1: Clown\n")
        f.write("artist=Cuttlefish That Loves Diving\n")
        f.write("album=Lord of Mysteries\n\n")
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
    print("\n🎬 Building M4B with ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ffmpeg failed:\n{result.stderr}")
    else:
        print(f"✅ M4B saved to: {output_path}")

    concat_list.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    CHAPTER_DIR.mkdir(parents=True, exist_ok=True)

    print("📖 Loading EPUB...")
    chapters = extract_text_from_epub(EPUB_PATH, page_start=PAGE_START, page_end=PAGE_END)
    print(f"   Found {len(chapters)} chapters (pages {PAGE_START}–{PAGE_END})")
    for i, (title, text) in enumerate(chapters):
        print(f"   [{i}] {title!r} — {len(text.split())} words")

    print("\n🧩 Splitting into chunks...")
    all_chunks = []
    chapter_chunk_ranges = {}
    global_idx = 0
    for ch_title, ch_text in chapters:
        chunks = split_into_chunks(ch_text, max_words=MAX_WORDS)
        start = global_idx
        for i, chunk in enumerate(chunks):
            all_chunks.append((ch_title, i, global_idx, chunk))
            global_idx += 1
        chapter_chunk_ranges[ch_title] = (start, global_idx - 1)
    print(f"   {len(chapters)} chapters → {len(all_chunks)} chunks")

    print("🤖 Loading Qwen3-TTS model...")
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        MODEL_CHOICE,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("   Model loaded!")

    existing_count = sum(1 for _, _, idx, _ in all_chunks if (CHUNK_DIR / f"chunk_{idx:04d}.wav").exists())
    remaining = len(all_chunks) - existing_count
    print(f"\n   {existing_count} chunks already done, {remaining} to generate")
    if remaining > 0:
        est = remaining * 9
        print(f"   ⏱  Estimated time: ~{est // 60}h {est % 60}m (at 9 min/chunk baseline)")

    chunk_files = {}
    failed_chunks = []
    chunk_times = []

    print("\n🎙️  Generating audio chunks...\n")
    pbar = tqdm(all_chunks, unit="chunk")

    for ch_title, ch_idx, g_idx, text in pbar:
        chunk_file = CHUNK_DIR / f"chunk_{g_idx:04d}.wav"
        chunk_files[g_idx] = chunk_file

        if chunk_file.exists():
            tqdm.write(f"⏭️  Skipping chunk {g_idx:04d} (already exists)")
            continue

        try:
            t0 = time.time()
            wav, sr = generate_with_retry(model, text, g_idx)
            elapsed = time.time() - t0
            chunk_times.append(elapsed)

            sf.write(str(chunk_file), wav, sr)

            avg = sum(chunk_times[-5:]) / len(chunk_times[-5:])
            remaining_now = sum(1 for _, _, i, _ in all_chunks if not (CHUNK_DIR / f"chunk_{i:04d}.wav").exists())
            eta_min = int(remaining_now * avg / 60)
            pbar.set_postfix({"last": f"{elapsed:.0f}s", "ETA": f"~{eta_min}m"})

            tqdm.write(f"✅ {g_idx:04d} | {ch_title} [{ch_idx}] | {len(text.split())} words | {elapsed:.1f}s")

        except Exception:
            tqdm.write(f"❌ Chunk {g_idx:04d} permanently failed — see {FAILURE_LOG.name}")
            failed_chunks.append(g_idx)

    if failed_chunks:
        print(f"\n⚠️  {len(failed_chunks)} chunks failed: {failed_chunks}")

    # ── Per-chapter FLACs ────────────────────────────────────────────────────
    print("\n📚 Stitching per-chapter files...")
    chapter_flac_files = []
    for ch_title, _ in chapters:
        start_idx, end_idx = chapter_chunk_ranges[ch_title]
        ch_chunks = [
            chunk_files[i] for i in range(start_idx, end_idx + 1)
            if i in chunk_files and chunk_files[i].exists()
        ]
        if not ch_chunks:
            print(f"   ⚠️  No chunks for {ch_title!r}, skipping")
            continue
        safe = re.sub(r'[^\w\s-]', '', ch_title).strip().replace(' ', '_')
        ch_flac = CHAPTER_DIR / f"{safe}.flac"
        duration = stitch_chapter(ch_chunks, ch_flac)
        chapter_flac_files.append((ch_title, ch_flac))
        print(f"   ✅ {ch_title} → {ch_flac.name} ({duration:.1f}s)")

    # ── M4B ──────────────────────────────────────────────────────────────────
    build_m4b(chapter_flac_files, FINAL_OUTPUT)


if __name__ == "__main__":
    main()
