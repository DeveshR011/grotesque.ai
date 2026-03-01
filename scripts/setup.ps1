<#
.SYNOPSIS
    Grotesque AI – Full Environment Setup (Windows)

.DESCRIPTION
    Installs Python venv, CUDA dependencies, builds llama-cpp-python
    with CUDA support, installs all requirements, and validates GPU.

.NOTES
    Run from project root as Administrator (for service install).
    Requires: Python 3.10-3.12, CUDA Toolkit 12.x, CMake, Visual Studio Build Tools.
#>

param(
    [switch]$SkipVenv,
    [switch]$SkipCuda,
    [switch]$SkipModels,
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent

Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║          Grotesque AI – Environment Setup            ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Set-Location $ProjectRoot

# ──────────────────────────────────────────────────────────────
# 1. Prerequisites check
# ──────────────────────────────────────────────────────────────
Write-Host "[1/8] Checking prerequisites…" -ForegroundColor Yellow

# Python
try {
    $pyVer = & $PythonPath --version 2>&1
    Write-Host "  ✓ $pyVer" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found. Install Python 3.10-3.12." -ForegroundColor Red
    exit 1
}

# CUDA
if (-not $SkipCuda) {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $cudaVer = & nvcc --version 2>&1 | Select-String "release"
        Write-Host "  ✓ CUDA: $cudaVer" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ nvcc not found. Install CUDA Toolkit 12.x" -ForegroundColor Yellow
        Write-Host "    https://developer.nvidia.com/cuda-downloads" -ForegroundColor Gray
    }

    # CMake
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmake) {
        Write-Host "  ✓ CMake found" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ CMake not found. Install from https://cmake.org" -ForegroundColor Yellow
    }
}

# nvidia-smi
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvsmi) {
    Write-Host "  ✓ nvidia-smi available" -ForegroundColor Green
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null |
        ForEach-Object { Write-Host "    GPU: $_" -ForegroundColor Gray }
} else {
    Write-Host "  ⚠ nvidia-smi not found" -ForegroundColor Yellow
}

# ──────────────────────────────────────────────────────────────
# 2. Create virtual environment
# ──────────────────────────────────────────────────────────────
if (-not $SkipVenv) {
    Write-Host ""
    Write-Host "[2/8] Creating virtual environment…" -ForegroundColor Yellow
    if (-not (Test-Path ".venv")) {
        & $PythonPath -m venv .venv
        Write-Host "  ✓ .venv created" -ForegroundColor Green
    } else {
        Write-Host "  ✓ .venv already exists" -ForegroundColor Green
    }
    # Activate
    & .\.venv\Scripts\Activate.ps1
    Write-Host "  ✓ Activated" -ForegroundColor Green

    # Upgrade pip
    & python -m pip install --upgrade pip setuptools wheel --quiet
    Write-Host "  ✓ pip upgraded" -ForegroundColor Green
}

# ──────────────────────────────────────────────────────────────
# 3. Install PyTorch (CUDA 12.x)
# ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/8] Installing PyTorch with CUDA…" -ForegroundColor Yellow
& pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
Write-Host "  ✓ PyTorch installed" -ForegroundColor Green

# ──────────────────────────────────────────────────────────────
# 4. Install llama-cpp-python with CUDA
# ──────────────────────────────────────────────────────────────
if (-not $SkipCuda) {
    Write-Host ""
    Write-Host "[4/8] Building llama-cpp-python with CUDA…" -ForegroundColor Yellow
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    $env:FORCE_CMAKE = "1"
    & pip install llama-cpp-python --force-reinstall --no-cache-dir --quiet
    Write-Host "  ✓ llama-cpp-python (CUDA) installed" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[4/8] Skipping CUDA build (--SkipCuda)" -ForegroundColor Yellow
}

# ──────────────────────────────────────────────────────────────
# 5. Install remaining dependencies
# ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[5/8] Installing Python dependencies…" -ForegroundColor Yellow
& pip install -r requirements.txt --quiet
Write-Host "  ✓ All requirements installed" -ForegroundColor Green

# ──────────────────────────────────────────────────────────────
# 6. Install CTranslate2 (for faster-whisper GPU)
# ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[6/8] Ensuring CTranslate2 CUDA support…" -ForegroundColor Yellow
& pip install ctranslate2 --quiet
Write-Host "  ✓ CTranslate2 ready" -ForegroundColor Green

# ──────────────────────────────────────────────────────────────
# 7. Download models
# ──────────────────────────────────────────────────────────────
if (-not $SkipModels) {
    Write-Host ""
    Write-Host "[7/8] Downloading models…" -ForegroundColor Yellow
    & python scripts\download_models.py
    Write-Host "  ✓ Models downloaded" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[7/8] Skipping model download (--SkipModels)" -ForegroundColor Yellow
}

# ──────────────────────────────────────────────────────────────
# 8. Validate GPU setup
# ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[8/8] Validating GPU setup…" -ForegroundColor Yellow

$validation = @"
import sys
print("Python:", sys.version)
try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import ctranslate2
    print(f"CTranslate2: {ctranslate2.__version__}")
except ImportError:
    print("CTranslate2: NOT INSTALLED")

try:
    from llama_cpp import Llama
    print("llama-cpp-python: OK")
except ImportError:
    print("llama-cpp-python: NOT INSTALLED")

try:
    from faster_whisper import WhisperModel
    print("faster-whisper: OK")
except ImportError:
    print("faster-whisper: NOT INSTALLED")

try:
    from piper import PiperVoice
    print("piper-tts: OK")
except ImportError:
    print("piper-tts: NOT INSTALLED")

try:
    import webrtcvad
    print("webrtcvad: OK")
except ImportError:
    print("webrtcvad: NOT INSTALLED")

try:
    import sounddevice
    print("sounddevice: OK")
except ImportError:
    print("sounddevice: NOT INSTALLED")
"@

& python -c $validation

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              Setup Complete!                         ║" -ForegroundColor Green
Write-Host "╠══════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║  Test:     python main.py --debug                    ║" -ForegroundColor Green
Write-Host "║  Service:  python main.py --install-service          ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
