# Grotesque AI – Deployment Guide

## Prerequisites

- **OS**: Windows 10/11 (primary), Linux (supported)
- **CPU**: Intel i7 13th Gen or equivalent (8+ threads)
- **GPU**: NVIDIA RTX 4050 (6GB VRAM) or better, CUDA 12.x
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 15 GB free on NVMe SSD
- **Python**: 3.10 – 3.12

## Windows Deployment

### Step 1: Environment Setup

```powershell
# Run the automated setup script (creates venv, installs CUDA deps)
.\scripts\setup.ps1
```

Or manually:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# llama-cpp-python with CUDA
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python==0.3.4 --force-reinstall --no-cache-dir

# All other dependencies
pip install -r requirements.txt
```

### Step 2: Download Models

```powershell
python scripts\download_models.py
```

This downloads:
- LLaMA 3 8B Instruct Q4_K_M (~4.6 GB)
- Faster-Whisper small (~400 MB)
- Piper TTS en_US-amy-medium (~60 MB)
- OpenWakeWord models (~5 MB)

### Step 3: Verify Installation

```powershell
python scripts\benchmark.py
```

Expected output:
- STT: < 200ms latency
- LLM: 300-450ms TTFT, 30+ tokens/sec
- TTS: < 100ms per sentence
- VRAM: < 5.5 GB total

### Step 4: Run as Foreground Process

```powershell
python main.py --debug
```

### Step 5: Install as Windows Service

```powershell
# Must be run as Administrator
python main.py --install-service

# Start the service
net start GrotesqueAI
```

### Step 6: Stress Test (Optional)

```powershell
# Short test (30 min)
python scripts\stress_test.py --duration-hours 0.5

# Full 24-hour test
python scripts\stress_test.py --duration-hours 24 --report stress_24h.json
```

## Linux Deployment (systemd)

### Create Service Unit

```ini
# /etc/systemd/system/grotesque-ai.service
[Unit]
Description=Grotesque AI Voice Assistant
After=sound.target network.target
Wants=sound.target

[Service]
Type=simple
User=grotesque
Group=audio
WorkingDirectory=/opt/grotesque-ai
ExecStart=/opt/grotesque-ai/.venv/bin/python main.py
Restart=always
RestartSec=5
Nice=-10
LimitCORE=0
LimitNOFILE=65535
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable grotesque-ai
sudo systemctl start grotesque-ai
```

## Troubleshooting

### CUDA Not Detected
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If False, reinstall PyTorch with CUDA support.

### VRAM Exceeded
Reduce `llm.n_gpu_layers` in config.yaml (e.g., 28 instead of 33).

### Audio Device Not Found
List devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
Set `audio.device_index` in config to the correct device number.

### Service Won't Start
Check Windows Event Viewer → Application logs for "GrotesqueAI" entries.
