"""
Grotesque AI – Model Downloader

Downloads all required models to the models/ directory.
Fully offline after this script completes.

Models:
 • LLM:  LLaMA 3 8B Instruct Q4_K_M GGUF (~4.7 GB)
 • STT:  Faster-Whisper small (~460 MB float16)
 • TTS:  Piper en_US-amy-medium (~75 MB)
 • Wake: OpenWakeWord pre-trained models (~5 MB)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress display."""
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading {desc or dest.name}…")
    print(f"    URL: {url}")

    try:
        urllib.request.urlretrieve(url, str(dest), _progress_hook)
        print(f"\n  ✓ Saved to {dest}")
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        if dest.exists():
            dest.unlink()
        raise


def _progress_hook(block_num, block_size, total_size):
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 / total_size)
        mb = block_num * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(
            f"\r    {pct:5.1f}%  [{mb:.0f}/{total_mb:.0f} MB]",
            end="",
            flush=True,
        )


def download_llm():
    """Download LLaMA 3 8B Instruct Q4_K_M GGUF."""
    print("\n[1/4] LLM: LLaMA 3 8B Instruct (Q4_K_M)")
    dest = MODELS_DIR / "llm" / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

    # HuggingFace bartowski quantization (popular, verified)
    url = (
        "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF"
        "/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    )

    print("  NOTE: This file is ~4.7 GB. Ensure sufficient disk space.")
    print("  If the automatic download fails, manually download from:")
    print(f"    {url}")
    print(f"  And place it at: {dest}")
    print()

    try:
        download_file(url, dest, "LLaMA 3 8B Q4_K_M")
    except Exception:
        print("\n  ⚠ Auto-download failed. Download manually:")
        print(f"    URL:  {url}")
        print(f"    Dest: {dest}")


def download_stt():
    """
    Faster-Whisper downloads models automatically via CTranslate2.
    We just create the directory and do a test import.
    """
    print("\n[2/4] STT: Faster-Whisper (auto-downloads on first use)")
    stt_dir = MODELS_DIR / "stt"
    stt_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Model directory: {stt_dir}")
    print("  ℹ The 'small' model (~460 MB) downloads on first run.")
    print("  To pre-download, run:")
    print('    python -c "from faster_whisper import WhisperModel; '
          "WhisperModel('small', device='cuda', compute_type='float16', "
          f"download_root='{stt_dir}')\"")


def download_tts():
    """Download Piper TTS voice model."""
    print("\n[3/4] TTS: Piper en_US-amy-medium")
    tts_dir = MODELS_DIR / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/"

    files = {
        "en_US-amy-medium.onnx": base_url + "en_US-amy-medium.onnx",
        "en_US-amy-medium.onnx.json": base_url + "en_US-amy-medium.onnx.json",
    }

    for filename, url in files.items():
        dest = tts_dir / filename
        try:
            download_file(url, dest, filename)
        except Exception:
            print(f"  ⚠ Failed to download {filename}. Download manually:")
            print(f"    URL:  {url}")
            print(f"    Dest: {dest}")


def download_wake():
    """
    OpenWakeWord models are bundled with the package.
    Create directory and note setup.
    """
    print("\n[4/4] Wake Word: OpenWakeWord")
    wake_dir = MODELS_DIR / "wake"
    wake_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Model directory: {wake_dir}")
    print("  ℹ OpenWakeWord ships with built-in models (hey_jarvis, etc.).")
    print("  No additional download needed for default keywords.")
    print("  For custom wake words, train with:")
    print("    https://github.com/dscripka/openWakeWord")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║       Grotesque AI – Model Downloader                ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Models directory: {MODELS_DIR}")

    download_llm()
    download_stt()
    download_tts()
    download_wake()

    print("\n" + "═" * 56)
    print("Model setup complete!")
    print()

    # Summary
    total_size = 0
    for f in MODELS_DIR.rglob("*"):
        if f.is_file() and f.name != ".gitkeep":
            total_size += f.stat().st_size
    print(f"Total model size: {total_size / (1024**3):.2f} GB")

    print("\nExpected VRAM usage at runtime:")
    print("  LLM (Q4_K_M, 33 layers on GPU): ~4.2 GB")
    print("  STT (Whisper small, float16):    ~0.5 GB")
    print("  TTS (Piper, CPU only):           ~0.0 GB")
    print("  ─────────────────────────────────────────")
    print("  Total VRAM:                      ~4.7 GB (< 5.5 GB target)")


if __name__ == "__main__":
    main()
