"""
Grotesque AI – Pipeline Orchestrator (v2)

Wires every component together and manages the full lifecycle:

  Mic → VAD → WakeWord → STT → LLM → IntentRouter → Plugin/TTS → Speaker

Phase 2 additions:
 • Intent Router between LLM and TTS
 • Plugin Engine with sandboxed built-in plugins
 • Memory System (short-term + optional FAISS vector)
 • Supervisor with heartbeat monitoring
 • Enhanced security (socket prevention, crash dump disable, AES logs)

All inter-thread communication goes through lock-free ring buffers
(audio hot-path) or bounded EventQueues (text / control messages).

Startup sequence:
  1. Load config
  2. Apply security hardening
  3. Pre-load all models (GPU + CPU)
  4. Allocate buffers
  5. Start threads in dependency order
  6. Start supervisor + watchdog
  7. Block on shutdown signal

Shutdown sequence (SIGINT / service stop):
  1. Signal all threads to stop
  2. Wait for graceful drain (configurable timeout)
  3. Save memory to disk (if vector enabled)
  4. Wipe all buffers
  5. Release GPU resources
  6. Remove firewall rule
  7. Exit 0
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

from core.audio_capture import AudioCapture
from core.audio_loopback import AudioLoopback
from core.audio_playback import AudioPlayback
from core.buffers import AudioRingBuffer, EventQueue
from core.intent_router import IntentRouter
from core.llm import LLMEngine, LLMRequest, LLMResponse, LLMToken
from core.memory import MemoryManager
from core.plugins.engine import PluginEngine
from core.security import (
    EncryptedLogHandler,
    apply_all_security,
    disable_core_dumps,
    hide_console_window,
    set_process_priority_high,
    setup_firewall,
    teardown_firewall,
)
from core.stt import SpeechToText, TranscriptionResult
from core.supervisor import Supervisor
from core.tts import TextToSpeech
from core.vad import VoiceActivityDetector
from core.wake_word import WakeWordEngine
from core.watchdog import Watchdog

logger = logging.getLogger("grotesque.pipeline")

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent


# ======================================================================
# Null stubs for text-only mode (no TTS / no audio playback)
# ======================================================================

class _NullTTS:
    """No-op TTS for text-only mode."""
    def __init__(self, token_queue=None, audio_queue=None, **kw):
        pass
    def load_model(self): pass
    def start(self): pass
    def stop(self): pass
    is_alive = property(lambda self: True)
    def set_heartbeat(self, m): pass


class _NullPlayback:
    """No-op audio playback for text-only mode."""
    def __init__(self, audio_queue=None, **kw):
        pass
    def start(self): pass
    def stop(self): pass
    is_playing = property(lambda self: False)
    def abort_playback(self): pass
    def set_heartbeat(self, m): pass
    is_alive = property(lambda self: True)


class Pipeline:
    """
    Top-level orchestrator.  Instantiate, call ``run()`` and it blocks
    until shutdown is requested.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path = config_path or str(BASE_DIR / "config" / "config.yaml")
        self.cfg: dict = {}
        self._shutdown = threading.Event()

        # Components (initialised in _build)
        self._capture_ring: Optional[AudioRingBuffer] = None
        self._wake_ring: Optional[AudioRingBuffer] = None
        self._speech_q: Optional[EventQueue] = None
        self._text_q: Optional[EventQueue] = None
        self._llm_req_q: Optional[EventQueue] = None
        self._llm_tok_q: Optional[EventQueue] = None
        self._llm_resp_q: Optional[EventQueue] = None   # structured JSON responses
        self._tts_audio_q: Optional[EventQueue] = None

        self._audio_capture: Optional[AudioCapture] = None
        self._vad: Optional[VoiceActivityDetector] = None
        self._wake_word: Optional[WakeWordEngine] = None
        self._stt: Optional[SpeechToText] = None
        self._llm: Optional[LLMEngine] = None
        self._tts: Optional[TextToSpeech] = None
        self._playback: Optional[AudioPlayback] = None
        self._watchdog: Optional[Watchdog] = None

        # Phase 2 components
        self._intent_router: Optional[IntentRouter] = None
        self._plugin_engine: Optional[PluginEngine] = None
        self._memory: Optional[MemoryManager] = None
        self._supervisor: Optional[Supervisor] = None

        # Dispatcher thread (routes STT → LLM, manages wake state)
        self._dispatcher_thread: Optional[threading.Thread] = None
        # Intent dispatcher thread (routes LLM responses → plugins/TTS)
        self._intent_dispatcher_thread: Optional[threading.Thread] = None
        # Monitor window (set externally by TrayApp after build)
        self._monitor = None
        # Loopback (speaker capture) components – initialised only when enabled
        self._loopback_ring: Optional[AudioRingBuffer] = None
        self._loopback_speech_q: Optional[EventQueue] = None
        self._loopback_capture: Optional[AudioLoopback] = None
        self._loopback_vad: Optional[VoiceActivityDetector] = None
        self._loopback_enabled: bool = False

        # Timestamp of last assistant speech (used by follow-up detection)
        self._last_assistant_speak_time: float = 0.0

    # ==================================================================
    # Public API
    # ==================================================================

    def run(self) -> None:
        """Blocking entry point.  Loads, starts, and blocks until shutdown."""
        self._load_config()
        self._setup_logging()
        self._apply_security()
        self._build()
        self._preload_models()
        self._start_all()
        self._install_signals()

        logger.info("═══ Grotesque AI pipeline running (v2) ═══")
        self._block_until_shutdown()
        self._shutdown_all()
        logger.info("═══ Grotesque AI pipeline stopped ═══")

    def request_shutdown(self) -> None:
        logger.info("Shutdown requested")
        self._shutdown.set()

    # ==================================================================
    # Config
    # ==================================================================

    def _load_config(self) -> None:
        path = Path(self._config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f) or {}
            logger.info("Config loaded from %s", path)
        else:
            logger.warning("Config not found at %s, using defaults", path)
            self.cfg = {}

    def _c(self, *keys, default=None):
        """Navigate nested config keys."""
        node = self.cfg
        for k in keys:
            if isinstance(node, dict):
                node = node.get(k)
            else:
                return default
            if node is None:
                return default
        return node

    # ==================================================================
    # Logging
    # ==================================================================

    def _setup_logging(self) -> None:
        level_str = self._c("runtime", "log_level", default="WARNING")
        level = getattr(logging, level_str.upper(), logging.WARNING)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
        # Suppress noisy third-party loggers
        for name in ("faster_whisper", "ctranslate2", "llama_cpp", "httpx", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)

        # Add encrypted log handler if configured
        if self._c("security", "encrypt_logs", default=False):
            from core.security import generate_key
            key_path = BASE_DIR / "config" / ".log_key"
            if key_path.exists():
                key = key_path.read_bytes()
            else:
                key = generate_key(key_path)
            handler = EncryptedLogHandler(
                log_path=BASE_DIR / "logs" / "grotesque.enc.log",
                encryption_key=key,
            )
            handler.setLevel(logging.WARNING)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            ))
            logging.getLogger().addHandler(handler)

    # ==================================================================
    # Security
    # ==================================================================

    def _apply_security(self) -> None:
        apply_all_security(
            block_outbound=self._c("security", "block_outbound", default=True),
            prevent_sockets=self._c("security", "prevent_socket_binding", default=True),
            disable_crash_dumps=self._c("security", "disable_crash_dumps", default=True),
            firewall_rule_name=self._c("security", "firewall_rule_name", default="GrotesqueAI_Block"),
        )

    def _teardown_security(self) -> None:
        if self._c("security", "block_outbound", default=True):
            rule = self._c("security", "firewall_rule_name", default="GrotesqueAI_Block")
            teardown_firewall(rule)

    # ==================================================================
    # Build (allocate buffers + instantiate components)
    # ==================================================================

    def _build(self) -> None:
        sr = self._c("audio", "sample_rate", default=16_000)
        channels = self._c("audio", "channels", default=1)
        frame_ms = self._c("audio", "frame_duration_ms", default=20)
        ring_sec = self._c("audio", "ring_buffer_sec", default=30)

        # --- Buffers ---
        self._capture_ring = AudioRingBuffer(ring_sec, sr, channels)
        self._wake_ring = AudioRingBuffer(5, sr, channels)
        self._speech_q = EventQueue(maxsize=32)
        self._text_q = EventQueue(maxsize=64)
        self._llm_req_q = EventQueue(maxsize=16)
        self._llm_tok_q = EventQueue(maxsize=512)
        self._llm_resp_q = EventQueue(maxsize=32)      # structured LLM responses
        self._tts_audio_q = EventQueue(maxsize=64)

        # --- Audio Capture ---
        self._audio_capture = AudioCapture(
            ring_buffer=self._capture_ring,
            sample_rate=sr,
            channels=channels,
            frame_duration_ms=frame_ms,
            device_index=self._c("audio", "device_index"),
        )

        # --- VAD ---
        self._vad = VoiceActivityDetector(
            ring_buffer=self._capture_ring,
            speech_queue=self._speech_q,
            sample_rate=sr,
            frame_duration_ms=frame_ms,
            aggressiveness=self._c("vad", "aggressiveness", default=3),
            speech_pad_ms=self._c("vad", "speech_pad_ms", default=300),
            min_speech_ms=self._c("vad", "min_speech_ms", default=250),
            silence_limit_ms=self._c("vad", "silence_limit_ms", default=1200),
        )

        # --- Wake Word ---
        self._wake_word = WakeWordEngine(
            ring_buffer=self._wake_ring,
            engine=self._c("wake_word", "engine", default="openwakeword"),
            model_path=str(BASE_DIR / self._c("wake_word", "model_path", default="models/wake/")),
            keyword=self._c("wake_word", "keyword", default="hey_jarvis"),
            threshold=self._c("wake_word", "threshold", default=0.5),
            sample_rate=sr,
            on_wake=self._on_wake_word,
            porcupine_model_path=self._c("wake_word", "porcupine_model_path"),
            porcupine_keyword_path=self._c("wake_word", "porcupine_keyword_path"),
        )

        # --- STT ---
        self._stt = SpeechToText(
            speech_queue=self._speech_q,
            text_queue=self._text_q,
            model_size=self._c("stt", "model_size", default="medium"),
            device=self._c("stt", "device", default="cuda"),
            compute_type=self._c("stt", "compute_type", default="int8_float16"),
            beam_size=self._c("stt", "beam_size", default=3),
            language=self._c("stt", "language", default="en"),
            vad_filter=self._c("stt", "vad_filter", default=True),
            model_dir=str(BASE_DIR / self._c("stt", "model_dir", default="models/stt/")),
            sample_rate=sr,
            condition_on_previous=self._c("stt", "condition_on_previous_text", default=False),
            initial_prompt=self._c("stt", "initial_prompt"),
            word_timestamps=self._c("stt", "word_timestamps", default=True),
            hotwords=self._c("stt", "hotwords", default=[]),
            # loopback_speech_queue is set later once we know if loopback is enabled
        )

        # --- LLM ---
        structured = self._c("intent_router", "enabled", default=True)
        self._llm = LLMEngine(
            request_queue=self._llm_req_q,
            token_queue=self._llm_tok_q,
            response_queue=self._llm_resp_q if structured else None,
            model_path=str(BASE_DIR / self._c("llm", "model_path",
                           default="models/llm/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")),
            n_gpu_layers=self._c("llm", "n_gpu_layers", default=33),
            n_ctx=self._c("llm", "n_ctx", default=1024),
            n_batch=self._c("llm", "n_batch", default=512),
            n_threads=self._c("llm", "n_threads", default=8),
            use_mmap=self._c("llm", "use_mmap", default=True),
            use_mlock=self._c("llm", "use_mlock", default=True),
            flash_attn=self._c("llm", "flash_attn", default=True),
            seed=self._c("llm", "seed", default=-1),
            max_tokens=self._c("llm", "max_tokens", default=150),
            temperature=self._c("llm", "temperature", default=0.5),
            top_p=self._c("llm", "top_p", default=0.9),
            top_k=self._c("llm", "top_k", default=40),
            repeat_penalty=self._c("llm", "repeat_penalty", default=1.1),
            stop_tokens=self._c("llm", "stop_tokens", default=["<|eot_id|>", "<|end_of_text|>"]),
            system_prompt=self._c("llm", "system_prompt",
                                  default="You are a helpful voice assistant."),
            structured_output=structured,
        )

        # --- TTS ---
        tts_model = self._c("tts", "model_path", default="models/tts/en_US-amy-medium.onnx")
        tts_config = self._c("tts", "config_path", default="models/tts/en_US-amy-medium.onnx.json")
        self._tts = TextToSpeech(
            token_queue=self._llm_tok_q,
            audio_queue=self._tts_audio_q,
            model_path=str(BASE_DIR / tts_model),
            config_path=str(BASE_DIR / tts_config),
            speaker_id=self._c("tts", "speaker_id", default=0),
            length_scale=self._c("tts", "length_scale", default=0.85),
            noise_scale=self._c("tts", "noise_scale", default=0.667),
            noise_w=self._c("tts", "noise_w", default=0.8),
            sentence_silence_sec=self._c("tts", "sentence_silence_sec", default=0.2),
            use_cuda=self._c("tts", "use_cuda", default=False),
            output_sample_rate=self._c("audio", "playback_sample_rate", default=22050),
            leading_silence_sec=self._c("tts", "leading_silence_sec", default=0.05),
        )

        # --- Playback ---
        self._playback = AudioPlayback(
            audio_queue=self._tts_audio_q,
            sample_rate=self._c("audio", "playback_sample_rate", default=22050),
            channels=self._c("audio", "playback_channels", default=1),
            device_index=self._c("audio", "playback_device_index"),
        )

        # --- Plugin Engine ---
        self._plugin_engine = PluginEngine(
            plugin_dirs=[str(BASE_DIR / "core" / "plugins" / "builtin")],
            config=self._c("plugins", default={}),
        )
        self._plugin_engine.discover_and_load()

        # --- Intent Router ---
        self._intent_router = IntentRouter(
            plugin_engine=self._plugin_engine,
            fallback_to_conversation=True,
        )

        # --- Memory System ---
        self._memory = MemoryManager(
            short_term_size=self._c("memory", "short_term_size", default=100),
            enable_vector=self._c("memory", "enable_vector", default=False),
            vector_index_path=(
                Path(BASE_DIR / self._c("memory", "vector_index_path", default="data/memory.idx"))
                if self._c("memory", "enable_vector", default=False) else None
            ),
        )

        # --- Watchdog ---
        self._watchdog = Watchdog(
            check_interval_sec=self._c("runtime", "watchdog_interval_sec", default=5),
            idle_clear_timeout_sec=self._c("security", "idle_clear_timeout_sec", default=5),
        )

        # --- Supervisor ---
        self._supervisor = Supervisor(
            check_interval_sec=self._c("supervisor", "check_interval_sec", default=10),
            on_gpu_fallback=self._on_gpu_fallback,
            on_fatal=self._on_fatal_error,
            gpu_max_vram_mb=self._c("supervisor", "gpu_max_vram_mb", default=7000),
            gpu_max_temperature=self._c("supervisor", "gpu_max_temperature", default=90),
        )

        # --- Loopback (speaker capture) ---
        self._loopback_enabled = self._c("audio", "loopback_enabled", default=False)
        if self._loopback_enabled:
            lb_ring_sec = self._c("audio", "loopback_ring_buffer_sec", default=10)
            self._loopback_ring = AudioRingBuffer(lb_ring_sec, sr, channels)
            self._loopback_speech_q = EventQueue(maxsize=32)

            self._loopback_capture = AudioLoopback(
                ring_buffer=self._loopback_ring,
                sample_rate=sr,
                channels=channels,
                frame_duration_ms=frame_ms,
                device_index=self._c("audio", "loopback_device_index"),
            )

            self._loopback_vad = VoiceActivityDetector(
                ring_buffer=self._loopback_ring,
                speech_queue=self._loopback_speech_q,
                sample_rate=sr,
                frame_duration_ms=frame_ms,
                aggressiveness=self._c("vad", "loopback_aggressiveness", default=2),
                speech_pad_ms=self._c("vad", "speech_pad_ms", default=300),
                min_speech_ms=self._c("vad", "min_speech_ms", default=250),
                silence_limit_ms=self._c("vad", "silence_limit_ms", default=1200),
            )
            logger.info("Loopback capture enabled (ring=%ds)", lb_ring_sec)

            # Inject loopback queue into STT so it polls both sources
            self._stt._loopback_q = self._loopback_speech_q

        # --- Text-only mode: replace TTS + Playback with null stubs ---
        if not self._c("gui", "audio_output", default=True):
            self._tts = _NullTTS(
                token_queue=self._llm_tok_q,
                audio_queue=self._tts_audio_q,
            )
            self._playback = _NullPlayback(
                audio_queue=self._tts_audio_q,
            )
            logger.info("Audio output disabled (text-only mode)")

        logger.info("Pipeline built successfully (v2 with intent routing + memory)")

    # ==================================================================
    # Model pre-loading
    # ==================================================================

    def _preload_models(self) -> None:
        logger.info("Pre-loading models…")
        t0 = time.monotonic()

        # GPU models
        self._stt.load_model()
        self._llm.load_model()

        # CPU models
        self._tts.load_model()

        elapsed = time.monotonic() - t0
        logger.info("All models loaded in %.1f s", elapsed)

    # ==================================================================
    # Start / stop threads
    # ==================================================================

    def _start_all(self) -> None:
        structured = self._c("intent_router", "enabled", default=True)

        # Start in dependency order
        self._audio_capture.start()
        self._vad.start()
        self._start_wake_feeder()
        self._wake_word.start()
        self._stt.start()
        self._llm.start()

        # TTS always needs to run – in structured mode it is fed by the
        # intent dispatcher; in unstructured mode by the LLM directly.
        self._tts.start()

        self._playback.start()

        # Start loopback capture if enabled
        if self._loopback_enabled and self._loopback_capture:
            self._loopback_capture.start()
            self._loopback_vad.start()

        # Check if loopback actually started (pyaudiowpatch may be missing)
        self._loopback_running = (
            self._loopback_enabled
            and self._loopback_capture is not None
            and self._loopback_capture.is_alive
        )

        # Dispatcher routes STT output → LLM with wake-word gating
        self._start_dispatcher()

        # Intent dispatcher routes LLM structured responses → plugins/TTS
        if structured:
            self._start_intent_dispatcher()

        # Register components with watchdog
        self._watchdog.register("AudioCapture", self._audio_capture, self._audio_capture.start)
        self._watchdog.register("VAD", self._vad, self._vad.start)
        self._watchdog.register("WakeWord", self._wake_word, self._wake_word.start)
        self._watchdog.register("STT", self._stt, self._stt.start)
        self._watchdog.register("LLM", self._llm, self._llm.start)
        self._watchdog.register("TTS", self._tts, self._tts.start)
        self._watchdog.register("Playback", self._playback, self._playback.start)

        # Loopback watchdog registration (only if actually running)
        if self._loopback_running:
            self._watchdog.register("LoopbackCapture", self._loopback_capture, self._loopback_capture.start)
            self._watchdog.register("LoopbackVAD", self._loopback_vad, self._loopback_vad.start)

        self._watchdog.register_wipeable(self._capture_ring)
        self._watchdog.register_wipeable(self._wake_ring)
        self._watchdog.register_wipeable(self._speech_q)
        self._watchdog.register_wipeable(self._text_q)
        self._watchdog.register_wipeable(self._llm_req_q)
        self._watchdog.register_wipeable(self._llm_tok_q)
        self._watchdog.register_wipeable(self._llm_resp_q)
        self._watchdog.register_wipeable(self._tts_audio_q)

        # Loopback wipeable buffers
        if self._loopback_running and self._loopback_ring:
            self._watchdog.register_wipeable(self._loopback_ring)
            self._watchdog.register_wipeable(self._loopback_speech_q)

        self._watchdog.register_wipe_callback(self._llm.clear_history)
        # NOTE: do NOT wipe VAD state on idle – it drops in-flight utterances
        self._watchdog.register_wipe_callback(self._memory.clear)

        self._watchdog.start()

        # Register components with supervisor heartbeat
        self._supervisor.register_component("AudioCapture", self._audio_capture.start, heartbeat_timeout=15.0)
        self._supervisor.register_component("VAD", self._vad.start, heartbeat_timeout=15.0)
        self._supervisor.register_component("STT", self._stt.start, heartbeat_timeout=30.0)
        self._supervisor.register_component("LLM", self._llm.start, heartbeat_timeout=60.0)
        self._supervisor.register_component("Playback", self._playback.start, heartbeat_timeout=15.0)

        # Loopback supervisor registration (only if actually running)
        if self._loopback_running:
            self._supervisor.register_component("LoopbackCapture", self._loopback_capture.start, heartbeat_timeout=60.0)
            self._supervisor.register_component("LoopbackVAD", self._loopback_vad.start, heartbeat_timeout=60.0)

        # Inject heartbeat monitor into components so they can send beats
        self._audio_capture.set_heartbeat(self._supervisor.heartbeat)
        self._vad.set_heartbeat(self._supervisor.heartbeat)
        self._stt.set_heartbeat(self._supervisor.heartbeat)
        self._llm.set_heartbeat(self._supervisor.heartbeat)
        self._playback.set_heartbeat(self._supervisor.heartbeat)

        # Loopback heartbeat injection
        if self._loopback_running:
            self._loopback_capture.set_heartbeat(self._supervisor.heartbeat)
            self._loopback_vad.set_heartbeat(self._supervisor.heartbeat)

        self._supervisor.start()

        # Notify monitor that pipeline is running
        if self._monitor:
            self._monitor.post("system", "Pipeline running")

    def _shutdown_all(self) -> None:
        timeout = self._c("runtime", "graceful_shutdown_timeout_sec", default=10)
        logger.info("Shutting down (timeout=%ds)…", timeout)

        # Stop supervisor first
        if self._supervisor:
            self._supervisor.stop()

        # Stop in reverse order
        self._watchdog.stop()
        self._playback.stop()
        self._tts.stop()
        self._llm.stop()
        self._stt.stop()
        self._wake_word.stop()
        self._vad.stop()
        self._audio_capture.stop()

        # Stop loopback components
        if self._loopback_enabled:
            if self._loopback_vad:
                self._loopback_vad.stop()
            if self._loopback_capture:
                self._loopback_capture.stop()

        # Save memory before wiping
        if self._memory:
            self._memory.save()
            logger.info("Memory saved: %s", self._memory.get_stats())
            self._memory.clear()

        # Security wipe all buffers
        self._capture_ring.clear()
        self._wake_ring.clear()
        self._speech_q.clear()
        self._text_q.clear()
        self._llm_req_q.clear()
        self._llm_tok_q.clear()
        self._llm_resp_q.clear()
        self._tts_audio_q.clear()
        self._llm.clear_history()

        # Wipe loopback buffers
        if self._loopback_ring:
            self._loopback_ring.clear()
        if self._loopback_speech_q:
            self._loopback_speech_q.clear()

        # Cleanup plugins
        if self._plugin_engine:
            self._plugin_engine.shutdown()

        self._teardown_security()

    # ==================================================================
    # Wake word feeder (copies capture ring → wake ring)
    # ==================================================================

    def _start_wake_feeder(self) -> None:
        """
        The capture ring is consumed by VAD.  Wake word needs its own copy.
        This light thread copies frames into the wake ring buffer.
        """
        def _feed():
            sr = self._c("audio", "sample_rate", default=16_000)
            frame_ms = self._c("audio", "frame_duration_ms", default=20)
            frame_size = int(sr * frame_ms / 1000)
            while not self._shutdown.is_set():
                frame = self._capture_ring.peek(frame_size)
                if frame is not None and len(frame) == frame_size:
                    self._wake_ring.write(frame)
                time.sleep(frame_ms / 1000 * 0.8)

        t = threading.Thread(target=_feed, name="WakeFeeder", daemon=True)
        t.start()

    # ==================================================================
    # Dispatcher (STT output → LLM requests with wake gating)
    # ==================================================================

    def _start_dispatcher(self) -> None:
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop, name="Dispatcher", daemon=True,
        )
        self._dispatcher_thread.start()

    def _dispatcher_loop(self) -> None:
        """
        Routes completed transcriptions to the LLM.

        Supports three modes (config: wake_word.mode):
          • always_on   – every MIC utterance goes to LLM (no wake word needed)
          • wake_word   – classic wake-word gate + timed follow-up window
          • push_to_talk – only active while hotkey is held

        Speaker (loopback) transcriptions always bypass gating and are
        stored as context.  Recent [SPEAKER] lines are prepended as a
        system message so the LLM knows what the PC is playing.
        """
        import re

        mode = self._c("wake_word", "mode", default="always_on")
        wake_timeout = self._c("wake_word", "timeout_sec", default=15.0)
        follow_up_sec = self._c("wake_word", "follow_up_sec", default=30.0)
        ptt_hotkey = self._c("wake_word", "push_to_talk_hotkey", default="ctrl+space")

        wake_active = (mode == "always_on")
        wake_time = time.monotonic() if wake_active else 0.0

        # Follow-up detection regex: short queries that reference context
        _FOLLOWUP_RE = re.compile(
            r"^(and |also |what about |how about |tell me more|go on|"
            r"then |so |yes|no|ok|okay|really|why|how|repeat that|"
            r"can you |could you |please )",
            re.IGNORECASE,
        )

        # Push-to-talk: try to load keyboard module
        ptt_pressed = False
        _keyboard = None
        if mode == "push_to_talk":
            try:
                import keyboard as _keyboard
                logger.info("Push-to-talk mode enabled (hotkey: %s)", ptt_hotkey)
            except ImportError:
                logger.warning("keyboard module not installed – falling back to wake_word mode")
                mode = "wake_word"

        def _is_followup(text: str) -> bool:
            """Check if text looks like a follow-up question."""
            return bool(_FOLLOWUP_RE.match(text)) or len(text.split()) <= 4

        def _should_pass(text: str) -> bool:
            """Determine whether a MIC transcription should reach the LLM."""
            nonlocal wake_active, wake_time, ptt_pressed

            if mode == "always_on":
                return True

            if mode == "push_to_talk" and _keyboard is not None:
                return _keyboard.is_pressed(ptt_hotkey)

            # wake_word mode
            if wake_active:
                return True

            # Follow-up window: if assistant recently spoke and this
            # looks like a short follow-up, pass it through
            if (self._last_assistant_speak_time > 0 and
                    (time.monotonic() - self._last_assistant_speak_time) < follow_up_sec and
                    _is_followup(text)):
                logger.info("Follow-up detected (%.0fs after assistant)",
                            time.monotonic() - self._last_assistant_speak_time)
                wake_active = True
                wake_time = time.monotonic()
                return True

            return False

        def _get_speaker_context_messages() -> list[dict[str, str]]:
            """Fetch recent [SPEAKER] context and format as a system message."""
            if not self._memory:
                return []
            speaker_entries = self._memory.get_context(n=5, role_filter="context")
            if not speaker_entries:
                return []
            # Only include entries from the last 30 seconds worth of context
            lines = [e["content"] for e in speaker_entries]
            if not lines:
                return []
            context_block = "\n".join(lines[-5:])
            return [{"role": "system", "content":
                      f"Recent audio playing through PC speakers:\n{context_block}"}]

        while not self._shutdown.is_set():
            # Check wake word activation (relevant for wake_word mode)
            if self._wake_word.activated:
                wake_active = True
                wake_time = time.monotonic()
                self._wake_word.clear_activation()
                self._watchdog.report_activity()
                logger.info("Wake active – listening for command (mode=%s)", mode)

                # Barge-in: stop any current playback
                if self._playback.is_playing:
                    self._playback.abort_playback()

            # Check timeout (wake_word mode only)
            if mode == "wake_word" and wake_active and (time.monotonic() - wake_time > wake_timeout):
                wake_active = False
                logger.debug("Wake timeout – returning to sleep")

            # Get transcription
            result = self._text_q.get(timeout=0.2)
            if result is None:
                continue

            if not isinstance(result, TranscriptionResult):
                continue

            # Only process final (non-partial) results
            if result.is_partial:
                continue

            text = result.text.strip()
            if not text:
                continue

            source = getattr(result, "source", "mic")

            # ---- Speaker (loopback) transcriptions ----
            if source == "speaker":
                logger.info("[SPEAKER] %s", text[:200])
                if self._memory:
                    self._memory.store_context(text=f"[SPEAKER] {text}")
                if self._monitor:
                    self._monitor.post("speaker", text)
                continue

            # ---- Microphone transcriptions ----
            if _should_pass(text):
                self._watchdog.report_activity()

                # Send heartbeat pulse
                if self._supervisor:
                    self._supervisor.heartbeat.beat("STT")

                logger.info("User said: %s", text[:200])
                if self._monitor:
                    self._monitor.post("user", text)

                # Build memory context: conversation history + recent speaker context
                memory_context = []
                if self._memory:
                    # Inject recent speaker audio as a system message
                    memory_context.extend(_get_speaker_context_messages())
                    # Then normal conversation context
                    memory_context.extend(self._memory.get_context(n=5))

                self._llm_req_q.put(LLMRequest(
                    user_text=text,
                    memory_context=memory_context,
                ))
                # Reset wake timer for multi-turn
                wake_time = time.monotonic()
            else:
                logger.debug("Ignoring speech (no activation): %s", text[:80])

    # ==================================================================
    # Intent Dispatcher (LLM responses → IntentRouter → Plugins/TTS)
    # ==================================================================

    def _start_intent_dispatcher(self) -> None:
        self._intent_dispatcher_thread = threading.Thread(
            target=self._intent_dispatcher_loop, name="IntentDispatcher", daemon=True,
        )
        self._intent_dispatcher_thread.start()

    def _intent_dispatcher_loop(self) -> None:
        """
        In structured output mode:
        1. Read LLMResponse from llm_resp_q
        2. Route through IntentRouter
        3. If plugin handled it, send plugin result text to TTS
        4. If conversation, send spoken_response to TTS
        5. Store exchange in memory
        6. Record assistant-speak time for follow-up window
        """
        while not self._shutdown.is_set():
            resp = self._llm_resp_q.get(timeout=0.2)
            if resp is None:
                continue
            if not isinstance(resp, LLMResponse):
                continue

            # Send heartbeat pulse
            if self._supervisor:
                self._supervisor.heartbeat.beat("LLM")

            # Determine the text to speak
            spoken_text = resp.spoken_response or resp.raw_text

            # Route through intent router
            if self._intent_router and resp.raw_text:
                try:
                    router_result = self._intent_router.route(resp.raw_text)
                    if router_result:
                        # Use the router's spoken_text (may include plugin overrides)
                        if router_result.spoken_text:
                            spoken_text = router_result.spoken_text
                        if router_result.intent:
                            logger.info(
                                "Intent '%s' routed (text: %s)",
                                router_result.intent.intent, spoken_text[:100],
                            )
                except Exception:
                    logger.exception("Intent routing error")

            # Post assistant response to monitor
            if self._monitor and spoken_text:
                self._monitor.post("assistant", spoken_text)

            # Feed spoken text to TTS as tokens
            if spoken_text:
                # Split into words for TTS streaming
                words = spoken_text.split()
                for i, word in enumerate(words):
                    token_text = word + (" " if i < len(words) - 1 else "")
                    self._llm_tok_q.put(LLMToken(text=token_text, is_final=False))
                self._llm_tok_q.put(LLMToken(text="", is_final=True))

            # Record when the assistant finishes speaking (for follow-up window)
            self._last_assistant_speak_time = time.monotonic()

            # Store in memory
            if self._memory:
                self._memory.store_exchange(
                    user_text=resp.user_text,
                    assistant_text=spoken_text,
                )

    def _on_wake_word(self) -> None:
        """Callback from wake word engine."""
        logger.debug("Wake word callback fired")

    def _on_gpu_fallback(self) -> None:
        """Called by supervisor when GPU becomes unavailable."""
        logger.critical("GPU fallback triggered – attempting to continue with reduced capability")

    def _on_fatal_error(self) -> None:
        """Called by supervisor on unrecoverable failure."""
        logger.critical("Fatal error detected by supervisor – requesting shutdown")
        self.request_shutdown()

    # ==================================================================
    # Signal handling
    # ==================================================================

    def _install_signals(self) -> None:
        def _handler(signum, frame):
            self.request_shutdown()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _block_until_shutdown(self) -> None:
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=1.0)
