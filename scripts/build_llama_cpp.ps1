<#
.SYNOPSIS
    Build llama.cpp with CUDA support from source (optional advanced setup).

.DESCRIPTION
    If the pip install with CMAKE_ARGS fails, this script builds
    llama-cpp-python from source with full CUDA support.

.NOTES
    Prerequisites:
    - CUDA Toolkit 12.x
    - CMake 3.21+
    - Visual Studio Build Tools 2022 (C++ workload)
    - Git
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent

Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    Build llama-cpp-python with CUDA (from source)    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan

# Check prereqs
$checkCmds = @("git", "cmake", "nvcc")
foreach ($cmd in $checkCmds) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "  ✓ $cmd found" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $cmd NOT found" -ForegroundColor Red
        Write-Host "    Install $cmd and ensure it's in PATH." -ForegroundColor Yellow
        exit 1
    }
}

# Activate venv if exists
if (Test-Path "$ProjectRoot\.venv\Scripts\Activate.ps1") {
    & "$ProjectRoot\.venv\Scripts\Activate.ps1"
}

# Method 1: pip with CUDA flags (preferred)
Write-Host ""
Write-Host "Attempting pip install with CUDA…" -ForegroundColor Yellow

$env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"  # Ada Lovelace (RTX 4050)
$env:FORCE_CMAKE = "1"

try {
    & pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose
    Write-Host "  ✓ llama-cpp-python built with CUDA!" -ForegroundColor Green
    Write-Host ""

    # Verify
    $verify = @"
from llama_cpp import Llama
print("llama-cpp-python import: OK")
# Check if CUDA/cuBLAS was linked
import llama_cpp
print(f"Backend: {getattr(llama_cpp, 'LLAMA_BACKEND', 'default')}")
"@
    & python -c $verify
    exit 0
}
catch {
    Write-Host "  ⚠ pip method failed. Trying source build…" -ForegroundColor Yellow
}

# Method 2: Clone and build from source
Write-Host ""
Write-Host "Building from source…" -ForegroundColor Yellow

$buildDir = "$ProjectRoot\build\llama-cpp-python"
if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}

New-Item -ItemType Directory -Force -Path "$ProjectRoot\build" | Out-Null
Set-Location "$ProjectRoot\build"

& git clone --recursive https://github.com/abetlen/llama-cpp-python.git
Set-Location $buildDir

$env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
$env:FORCE_CMAKE = "1"

& pip install -e . --verbose

Write-Host ""
Write-Host "  ✓ llama-cpp-python built from source!" -ForegroundColor Green

Set-Location $ProjectRoot
