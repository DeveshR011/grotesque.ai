# Grotesque AI – Architecture & Technical Documentation

## Overview

Grotesque AI is a **fully local, real-time, GPU-accelerated voice assistant** that runs 100% offline as a background service. Zero telemetry, zero cloud dependency – all data stays on-device.

**Target Hardware:** Intel i7 13th Gen HX + RTX 4050 Mobile (6GB VRAM) + 16GB+ RAM + NVMe SSD

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        GROTESQUE AI v2                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐  ┌──────┐  ┌───────────┐  ┌──────┐                │
│  │ Mic     │→ │ VAD  │→ │ Wake Word │→ │ STT  │                │
│  │ Capture │  │ (T2) │  │   (T3)    │  │ (T4) │                │
│  │  (T1)   │  └──────┘  └───────────┘  └──┬───┘                │
│  └─────────┘                               │                    │
│                                            ▼                    │
│                               ┌─────────────────────┐           │
│                               │   LLM Engine (T5)   │           │
│                               │  LLaMA 3 8B Q4_K_M  │           │
│                               │  Structured JSON Out │           │
│                               └──────────┬──────────┘           │
│                                          │                      │
│                               ┌──────────▼──────────┐           │
│                               │   Intent Router     │           │
│                               │  (Parse + Dispatch)  │           │
│                               └───┬──────────┬──────┘           │
│                                   │          │                  │
│                          ┌────────▼──┐  ┌────▼───────┐          │
│                          │  Plugin   │  │  TTS (T6)  │          │
│                          │  Engine   │  │  Piper CPU │          │
│                          │ (Sandbox) │  └──────┬─────┘          │
│                          └───────────┘         │                │
│                                          ┌─────▼─────┐          │
│                                          │  Speaker  │          │
│                                          │ Playback  │          │
│                                          └───────────┘          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │           Support Services                                │    │
│  │  ┌──────────┐  ┌────────────┐  ┌───────────┐            │    │
│  │  │ Watchdog │  │ Supervisor │  │  Memory   │            │    │
│  │  │   (T7)   │  │ Heartbeat  │  │  Manager  │            │    │
│  │  └──────────┘  └────────────┘  └───────────┘            │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │           Security Layer                                  │    │
│  │  Firewall | Socket Block | Crash Dump Off | AES Logs     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Thread Model

| Thread | Module | Hardware | Responsibility |
|--------|--------|----------|----------------|
| T1 | `audio_capture.py` | CPU | Mic → Ring Buffer (16kHz int16) |
| T2 | `vad.py` | CPU | WebRTC VAD, speech detection |
| T3 | `wake_word.py` | CPU | OpenWakeWord / Porcupine |
| T4 | `stt.py` | **GPU** | Faster-Whisper (CTranslate2 CUDA) |
| T5 | `llm.py` | **GPU** | LLaMA 3 8B Q4_K_M (llama-cpp CUDA) |
| T6 | `tts.py` + `audio_playback.py` | CPU | Piper TTS + sounddevice output |
| T7 | `watchdog.py` | CPU | Health monitoring, auto-restart |
| T8 | `supervisor.py` | CPU | Heartbeat, GPU fallback, leak detect |
| – | `pipeline.py` (Dispatcher) | CPU | Wake gating, STT→LLM routing |
| – | `pipeline.py` (IntentDispatcher) | CPU | LLM→IntentRouter→Plugin/TTS |

---

## Data Flow (Structured Output Mode)

1. **Mic → VAD**: Audio frames → WebRTC VAD detects speech segments
2. **VAD → STT**: Speech segments queued for transcription
3. **STT → Dispatcher**: Final transcription with confidence
4. **Dispatcher → LLM**: LLMRequest (with memory context injected)
5. **LLM → Intent Dispatcher**: LLMResponse with structured JSON:
   ```json
   {"intent": "set_timer", "parameters": {"duration_seconds": 300}, "response": "Timer set for 5 minutes"}
   ```
6. **Intent Dispatcher → IntentRouter**: Route to plugin or TTS
7. **Plugin/TTS → Speaker**: Audio output with barge-in support

---

## VRAM Budget

| Component | VRAM |
|-----------|------|
| LLaMA 3 8B Q4_K_M (33 layers) | ~4.2 GB |
| Faster-Whisper small (float16) | ~0.5 GB |
| **Total** | **~4.7 GB** |
| Available (RTX 4050 6GB) | 6.0 GB |
| **Headroom** | **~1.3 GB** |

---

## Performance Targets

| Metric | Target | Achieved By |
|--------|--------|-------------|
| STT Latency | < 200ms | Greedy decoding, beam_size=1 |
| LLM TTFT | 300-450ms | GPU offload, flash attention, Q4_K_M |
| LLM Throughput | > 30 tok/s | CUDA, batched KV cache |
| TTS Latency | < 100ms/sentence | Piper CPU, sentence streaming |
| End-to-end | < 1s wake-to-speech | Pipeline parallelism |

---

## Intent Router

The Intent Router sits between LLM output and TTS. The LLM returns structured JSON with an `intent` field. The router:

1. Parses JSON (handles markdown fences, malformed output)
2. Validates intent against registered handlers
3. Sanitizes parameters (strips forbidden chars, length limits)
4. Dispatches to plugin (sandboxed) or falls back to TTS

### Supported Intents

| Intent | Plugin | Example |
|--------|--------|---------|
| `conversation` | (none – direct to TTS) | "What's the weather?" |
| `get_time` | SystemQueryPlugin | "What time is it?" |
| `get_date` | SystemQueryPlugin | "What's today's date?" |
| `system_query` | SystemQueryPlugin | "How much battery left?" |
| `set_timer` | TimerAlarmPlugin | "Set a 5-minute timer" |
| `set_alarm` | TimerAlarmPlugin | "Wake me at 7 AM" |
| `media_control` | MediaControlPlugin | "Pause the music" |
| `change_setting` | SettingsPlugin | "Set volume to 80" |

---

## Plugin Architecture

### Creating a Plugin

```python
from core.plugins.engine import Plugin, PluginPermission

class MyPlugin(Plugin):
    name = "my_plugin"
    version = "1.0"
    intents = ["my_action"]
    permissions = PluginPermission.READ_SYSTEM_INFO

    def execute(self, intent: str, parameters: dict) -> dict:
        return {"result": "Done!", "speak": "Action completed."}

    def validate_parameters(self, intent: str, parameters: dict) -> bool:
        return True
```

### Sandbox Features

- **Timeout**: Configurable max execution time (default 10s)
- **Permissions**: Flag-based system (READ_SYSTEM_INFO, MEDIA_CONTROL, TIMER, etc.)
- **Input sanitization**: Forbidden characters stripped, length limits enforced
- **Audit logging**: All plugin executions logged with timestamps

---

## Memory System

### Phase 1: Short-term (RAM-only)
- Sliding window of recent exchanges (default: 100)
- No persistence, no disk I/O
- Auto-cleared on idle and shutdown
- Injected into LLM context for multi-turn conversations

### Phase 2: Vector Memory (FAISS)
- Local FAISS index for semantic search
- Embeddings via sentence-transformers (all-MiniLM-L6-v2)
- AES-256 encrypted persistence
- Fully detachable – system works without it
- Enable with `memory.enable_vector: true` in config

---

## Security Hardening

| Feature | Implementation |
|---------|---------------|
| Outbound firewall | Windows Firewall rule blocks process |
| Socket binding | Monkey-patched `socket.bind()` prevention |
| Crash dumps | Windows Error Reporting disabled |
| Memory wiping | `ctypes.memset` zeroing of buffers |
| Config encryption | Fernet (AES-256) for config files |
| Log encryption | AES-256 encrypted logging handler |
| Process priority | HIGH_PRIORITY_CLASS |
| Console hidden | `ShowWindow(SW_HIDE)` |
| Core dump disable | `RLIMIT_CORE = 0` (Linux) |
| Restricted user | Warning if running as admin |

---

## Windows Service

### Install
```powershell
# Run as Administrator
python main.py --install-service
```

### Configuration Applied Automatically:
- **Delayed auto-start**: Starts after critical system services
- **Recovery options**: Restart on 1st (5s), 2nd (10s), subsequent (30s) failures
- **High priority**: `HIGH_PRIORITY_CLASS`
- **No console window**: Hidden at service and process level

### Management
```powershell
net start GrotesqueAI     # Start
net stop GrotesqueAI      # Stop
sc query GrotesqueAI      # Status
```

---

## Project Structure

```
Grotesque ai/
├── config/
│   └── config.yaml              # Full configuration (v2)
├── core/
│   ├── __init__.py
│   ├── audio_capture.py         # T1: Mic → Ring Buffer
│   ├── audio_playback.py        # T6b: Audio output + barge-in
│   ├── buffers.py               # AudioRingBuffer + EventQueue
│   ├── intent_router.py         # Intent parsing + dispatch
│   ├── llm.py                   # T5: LLaMA 3 8B (structured JSON)
│   ├── memory.py                # Memory Manager (RAM + FAISS)
│   ├── pipeline.py              # Orchestrator (v2)
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── engine.py            # Plugin ABC + sandbox engine
│   │   └── builtin/
│   │       ├── __init__.py
│   │       ├── media_control.py # Media key simulation
│   │       ├── settings.py      # Runtime config changes
│   │       ├── system_query.py  # Time, date, battery, CPU
│   │       └── timer_alarm.py   # Timers and alarms
│   ├── security.py              # Full security hardening
│   ├── stt.py                   # T4: Faster-Whisper GPU
│   ├── supervisor.py            # Heartbeat + GPU fallback
│   ├── tts.py                   # T6a: Piper TTS synthesis
│   ├── vad.py                   # T2: WebRTC VAD
│   ├── wake_word.py             # T3: OpenWakeWord/Porcupine
│   └── watchdog.py              # T7: Health monitoring
├── models/
│   ├── llm/                     # LLaMA 3 8B GGUF
│   ├── stt/                     # Faster-Whisper models
│   ├── tts/                     # Piper ONNX models
│   └── wake/                    # Wake word models
├── scripts/
│   ├── benchmark.py             # Performance benchmarks
│   ├── build_llama_cpp.ps1      # CUDA build script
│   ├── download_models.py       # Model downloader
│   ├── setup.ps1                # Full environment setup
│   └── stress_test.py           # 24-hour stability tests
├── service/
│   ├── install_service.py       # Service CLI
│   └── windows_service.py       # pywin32 service (hardened)
├── main.py                      # Entry point
├── pyproject.toml
└── requirements.txt             # Dependencies (v2)
```

---

## Quick Start

```powershell
# 1. Run automated setup
.\scripts\setup.ps1

# 2. Download all models
python scripts\download_models.py

# 3. Run benchmarks
python scripts\benchmark.py

# 4. Start the assistant
python main.py --debug

# 5. Run stress tests (optional)
python scripts\stress_test.py --duration-hours 0.5

# 6. Install as service (admin)
python main.py --install-service
```

---

## Configuration Reference

See `config/config.yaml` for all options. Key sections:

| Section | Purpose |
|---------|---------|
| `runtime` | Device, logging, watchdog interval |
| `audio` | Sample rate, buffer sizes, device selection |
| `vad` | Aggressiveness, silence thresholds |
| `wake_word` | Engine selection, keyword, threshold |
| `stt` | Model size, compute type, beam size |
| `llm` | Model path, GPU layers, generation params |
| `tts` | Voice model, speed, noise settings |
| `intent_router` | Enable/disable structured output |
| `plugins` | Sandbox timeout, permission whitelist |
| `memory` | Short-term size, FAISS toggle, encryption |
| `supervisor` | Heartbeat timeout, VRAM limits |
| `security` | Firewall, socket block, crash dumps, AES logs |
