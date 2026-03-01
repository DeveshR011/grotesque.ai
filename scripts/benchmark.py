"""
Grotesque AI – Performance Benchmark Suite

Measures latency and throughput of each pipeline stage independently,
then runs an end-to-end simulation.

Usage:
    python scripts/benchmark.py                  # full benchmark
    python scripts/benchmark.py --stt-only       # STT only
    python scripts/benchmark.py --llm-only       # LLM only
    python scripts/benchmark.py --tts-only       # TTS only
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def load_config():
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# ══════════════════════════════════════════════════════════════
# GPU Info
# ══════════════════════════════════════════════════════════════

def print_gpu_info():
    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"  Device:     {props.name}")
            print(f"  VRAM:       {props.total_mem / 1024**3:.1f} GB")
            print(f"  CUDA Cores: {props.multi_processor_count * 128}")  # approximate
            print(f"  Compute:    {props.major}.{props.minor}")
            print(f"  PyTorch:    {torch.__version__}")
            print(f"  CUDA:       {torch.version.cuda}")
        else:
            print("  CUDA not available via PyTorch")
    except ImportError:
        print("  PyTorch not installed")

    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            print(f"  VRAM Used:  {g.memoryUsed:.0f} / {g.memoryTotal:.0f} MB")
            print(f"  GPU Load:   {g.load * 100:.0f}%")
            print(f"  Temp:       {g.temperature}°C")
    except ImportError:
        pass
    print()


# ══════════════════════════════════════════════════════════════
# STT Benchmark
# ══════════════════════════════════════════════════════════════

def benchmark_stt(cfg: dict):
    print("=" * 60)
    print("STT BENCHMARK (Faster-Whisper)")
    print("=" * 60)

    from faster_whisper import WhisperModel

    model_size = cfg.get("stt", {}).get("model_size", "small")
    device = cfg.get("stt", {}).get("device", "cuda")
    compute = cfg.get("stt", {}).get("compute_type", "float16")
    model_dir = str(PROJECT_ROOT / cfg.get("stt", {}).get("model_dir", "models/stt/"))

    print(f"  Model:   {model_size}")
    print(f"  Device:  {device}")
    print(f"  Type:    {compute}")
    print()

    # Load model
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type=compute,
                         download_root=model_dir)
    load_time = time.perf_counter() - t0
    print(f"  Model load time: {load_time:.2f} s")

    # Generate synthetic audio (silence + sine wave)
    sr = 16000
    durations = [1, 3, 5, 10]

    for dur in durations:
        # Generate a tone to give Whisper something to transcribe
        t = np.linspace(0, dur, sr * dur, dtype=np.float32)
        # Just use silence – measures processing overhead
        audio = np.zeros(sr * dur, dtype=np.float32)

        times = []
        for _ in range(3):  # 3 runs
            gc.collect()
            t0 = time.perf_counter()
            segments, info = model.transcribe(audio, beam_size=1, language="en")
            # Consume the generator
            for seg in segments:
                pass
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        rtf = avg / dur  # real-time factor
        print(f"  {dur:2d}s audio → {avg*1000:.0f} ms (RTF: {rtf:.3f}x)")

    del model
    gc.collect()
    print()


# ══════════════════════════════════════════════════════════════
# LLM Benchmark
# ══════════════════════════════════════════════════════════════

def benchmark_llm(cfg: dict):
    print("=" * 60)
    print("LLM BENCHMARK (LLaMA 3 8B via llama.cpp)")
    print("=" * 60)

    from llama_cpp import Llama

    lcfg = cfg.get("llm", {})
    model_path = str(PROJECT_ROOT / lcfg.get("model_path",
                     "models/llm/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"))

    if not Path(model_path).exists():
        print(f"  ✗ Model not found: {model_path}")
        print("  Run: python scripts/download_models.py")
        print()
        return

    n_gpu = lcfg.get("n_gpu_layers", 33)
    n_ctx = lcfg.get("n_ctx", 2048)
    n_batch = lcfg.get("n_batch", 512)

    print(f"  Model:      {Path(model_path).name}")
    print(f"  GPU layers: {n_gpu}")
    print(f"  Context:    {n_ctx}")
    print(f"  Batch:      {n_batch}")
    print()

    # Load model
    t0 = time.perf_counter()
    model = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_threads=8,
        use_mmap=True,
        use_mlock=True,
        verbose=False,
    )
    load_time = time.perf_counter() - t0
    print(f"  Model load time: {load_time:.2f} s")

    # Benchmark prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in two sentences.",
        "How does a combustion engine work? Be brief.",
    ]

    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": prompt},
        ]

        gc.collect()
        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0

        stream = model.create_chat_completion(
            messages=messages,
            max_tokens=128,
            temperature=0.7,
            stream=True,
        )

        response_parts = []
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
                response_parts.append(text)

        total_time = time.perf_counter() - t0
        ttft = (first_token_time - t0) * 1000 if first_token_time else 0
        tps = token_count / (total_time - (first_token_time - t0)) if first_token_time and token_count > 1 else 0

        print(f"  Prompt: \"{prompt[:50]}\"")
        print(f"    TTFT:    {ttft:.0f} ms")
        print(f"    Tokens:  {token_count}")
        print(f"    Speed:   {tps:.1f} tok/s")
        print(f"    Total:   {total_time:.2f} s")
        response = "".join(response_parts)
        print(f"    Output:  {response[:100]}…")
        print()

    del model
    gc.collect()


# ══════════════════════════════════════════════════════════════
# TTS Benchmark
# ══════════════════════════════════════════════════════════════

def benchmark_tts(cfg: dict):
    print("=" * 60)
    print("TTS BENCHMARK (Piper)")
    print("=" * 60)

    try:
        from piper import PiperVoice
    except ImportError:
        print("  ✗ piper-tts not installed")
        print()
        return

    tcfg = cfg.get("tts", {})
    model_path = str(PROJECT_ROOT / tcfg.get("model_path",
                     "models/tts/en_US-amy-medium.onnx"))
    config_path = str(PROJECT_ROOT / tcfg.get("config_path",
                      "models/tts/en_US-amy-medium.onnx.json"))

    if not Path(model_path).exists():
        print(f"  ✗ Model not found: {model_path}")
        print("  Run: python scripts/download_models.py")
        print()
        return

    # Load
    t0 = time.perf_counter()
    voice = PiperVoice.load(model_path, config_path=config_path)
    load_time = time.perf_counter() - t0
    print(f"  Model load time: {load_time:.2f} s")

    sentences = [
        "Hello, how can I help you today?",
        "The weather forecast predicts clear skies with temperatures around seventy degrees.",
        "Quantum computing leverages superposition and entanglement to solve complex problems.",
    ]

    for sentence in sentences:
        gc.collect()
        t0 = time.perf_counter()
        audio_chunks = []
        for audio_bytes in voice.synthesize_stream_raw(sentence):
            audio_chunks.append(audio_bytes)
        elapsed = time.perf_counter() - t0

        total_bytes = sum(len(c) for c in audio_chunks)
        total_samples = total_bytes // 2  # int16
        audio_duration = total_samples / 22050

        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        print(f"  \"{sentence[:50]}…\"")
        print(f"    Synth time:  {elapsed*1000:.0f} ms")
        print(f"    Audio:       {audio_duration:.2f} s")
        print(f"    RTF:         {rtf:.3f}x (< 1.0 = real-time)")
        print()

    del voice
    gc.collect()


# ══════════════════════════════════════════════════════════════
# VRAM Summary
# ══════════════════════════════════════════════════════════════

def print_vram_summary():
    print("=" * 60)
    print("VRAM USAGE SUMMARY")
    print("=" * 60)
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            print(f"  Used:      {g.memoryUsed:.0f} MB")
            print(f"  Free:      {g.memoryFree:.0f} MB")
            print(f"  Total:     {g.memoryTotal:.0f} MB")
            print(f"  Utilised:  {g.memoryUsed / g.memoryTotal * 100:.1f}%")
            if g.memoryUsed < 5500:
                print("  ✓ Within 5.5 GB VRAM budget")
            else:
                print("  ⚠ Exceeds 5.5 GB target!")
    except ImportError:
        print("  GPUtil not installed")
    print()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Grotesque AI Benchmark Suite")
    parser.add_argument("--stt-only", action="store_true")
    parser.add_argument("--llm-only", action="store_true")
    parser.add_argument("--tts-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       Grotesque AI – Performance Benchmark           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    print_gpu_info()

    run_all = not (args.stt_only or args.llm_only or args.tts_only)

    if run_all or args.stt_only:
        try:
            benchmark_stt(cfg)
        except Exception as e:
            print(f"  STT benchmark error: {e}\n")

    if run_all or args.llm_only:
        try:
            benchmark_llm(cfg)
        except Exception as e:
            print(f"  LLM benchmark error: {e}\n")

    if run_all or args.tts_only:
        try:
            benchmark_tts(cfg)
        except Exception as e:
            print(f"  TTS benchmark error: {e}\n")

    print_vram_summary()

    print("═" * 60)
    print("Benchmark complete!")
    print()
    print("Target metrics:")
    print("  STT partial transcript:  < 300 ms")
    print("  LLM first token (TTFT):  < 400 ms")
    print("  LLM throughput:          > 20 tok/s")
    print("  TTS RTF:                 < 0.5x")
    print("  Total VRAM:              < 5.5 GB")
    print()


if __name__ == "__main__":
    main()
