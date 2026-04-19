"""TTS provider layer.

Default provider is Piper (local/offline) and can be changed via .env:
    TTS_PROVIDER=piper | openai | pyttsx3

Callers keep using speak(text) with no API changes.

Piper performance architecture (persistent process + real-time streaming):
─────────────────────────────────────────────────────────────────────────
  OLD (per-call):  spawn → load model → generate → play → kill   (~800 ms overhead)
  NEW (persistent): spawn once → load model once, then per call:
                    write text → stream PCM chunks → play as they arrive

  This eliminates model-load latency on every utterance — critical on
  resource-constrained boards like Jetson Nano/Orin where loading the
  ONNX model can take 300–900 ms.

  Streaming architecture:
    • A dedicated reader thread drains piper's stdout into an audio queue
      as fast as piper produces bytes — playback starts on the FIRST chunk,
      not after full generation.
    • sounddevice RawOutputStream consumes the queue in real time.
    • A sentinel (None) in the queue signals end-of-utterance.

  Piper's stdio protocol (persistent mode):
    stdin  ← one UTF-8 line of text per utterance (newline-terminated)
    stdout → raw 16-bit LE PCM at PIPER_SAMPLE_RATE, no header
    stderr → log / error messages (monitored in a background thread)

  The persistent process survives across speak() calls. It is restarted
  automatically if it crashes. atexit() ensures clean shutdown.
"""

from __future__ import annotations

import atexit
import importlib
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_val: str, default_rel: str) -> Path:
    raw = env_val.strip() if env_val else ""
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else _PROJECT_ROOT / p
    return _PROJECT_ROOT / default_rel


# ── Config ──────────────────────────────────────────────────────────────────
# Audio queue chunk size in bytes (must be even — int16 samples).
# Smaller  → lower first-audio latency, more queue overhead.
# Larger   → smoother playback, slightly higher first-audio latency.
STREAM_CHUNK_BYTES = 2048

OPENAI_TTS_MODEL  = os.getenv("OPENAI_TTS_MODEL",  "tts-1")
OPENAI_TTS_VOICE  = os.getenv("OPENAI_TTS_VOICE",  "alloy")
OPENAI_TTS_SPEED  = float(os.getenv("OPENAI_TTS_SPEED", "1.0"))
OPENAI_TTS_FORMAT = "pcm"
OPENAI_TTS_RATE   = int(os.getenv("OPENAI_TTS_RATE", "24000"))

PIPER_EXE         = _resolve(os.getenv("PIPER_EXE",        ""), "models/piper/piper.exe")
PIPER_MODEL_PATH  = _resolve(os.getenv("PIPER_MODEL_PATH", ""), "models/piper/en_US-lessac-medium.onnx")
PIPER_SAMPLE_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))

_piper_config_env = os.getenv("PIPER_CONFIG_PATH", "").strip()
if _piper_config_env:
    PIPER_CONFIG_PATH: Path | None = _resolve(_piper_config_env, "")
else:
    _inferred = Path(str(PIPER_MODEL_PATH) + ".json")
    PIPER_CONFIG_PATH = _inferred if _inferred.exists() else None

PYTTSX3_RATE   = 170
PYTTSX3_VOLUME = 1.0

_WIN_DLL_MISSING = 3221225781   # 0xC0000135
# ────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  PERSISTENT PIPER PROCESS MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class _PiperProcess:
    """
    Manages a single long-lived piper subprocess.

    Lifecycle
    ---------
    start()   — spawn piper, launch reader + stderr threads
    speak()   — write text line; reader thread streams PCM to sounddevice
    shutdown()— gracefully terminate; called by atexit

    Thread safety
    -------------
    speak() acquires _lock so only one utterance plays at a time.
    The reader thread runs continuously, gating on _utterance_active event.
    """

    def __init__(self) -> None:
        self._proc:            subprocess.Popen | None = None
        self._lock             = threading.Lock()
        self._utterance_active = threading.Event()   # set while generating
        self._utterance_done   = threading.Event()   # set by stderr monitor when PCM is complete
        self._audio_q:         queue.Queue[bytes | None] = queue.Queue(maxsize=64)
        self._reader_thread:   threading.Thread | None = None
        self._stderr_thread:   threading.Thread | None = None
        self._ok               = False
        self._last_infer_sec:  float | None = None
        self._last_audio_sec:  float | None = None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        cmd = [
            str(PIPER_EXE),
            "--model",      str(PIPER_MODEL_PATH),
            "--output-raw",
            # Keep stdin open so we can send multiple lines
        ]
        if PIPER_CONFIG_PATH is not None and PIPER_CONFIG_PATH.exists():
            cmd.extend(["--config", str(PIPER_CONFIG_PATH)])
        return cmd

    def _reader_loop(self) -> None:
        """
        Background thread: drain piper stdout continuously.
        Chunks are placed into _audio_q while _utterance_active is set.
        A None sentinel is enqueued when piper's stdout goes silent
        (piper flushes after each input line).
        """
        assert self._proc is not None
        assert self._proc.stdout is not None

        while True:
            try:
                chunk = self._proc.stdout.read(STREAM_CHUNK_BYTES)
            except (OSError, ValueError):
                # Process died or pipe closed
                self._audio_q.put(None)
                break

            if not chunk:
                # EOF — piper exited
                self._audio_q.put(None)
                break

            if self._utterance_active.is_set():
                self._audio_q.put(chunk)

    def _stderr_monitor(self) -> None:
        """
        Background thread: read piper stderr line by line.

        Piper emits exactly one "Real-time factor:" line to stderr per utterance,
        written immediately after the last PCM byte is flushed to stdout.
        This is our precise end-of-utterance signal.

        From the observed logs:
            [piper] [info] Real-time factor: 0.104 (infer=1.23 sec, audio=11.76 sec)

        When we see this line, all PCM for the current utterance is already in
        the pipe — we set _utterance_done so speak() can exit the playback loop
        immediately instead of waiting for a 1.2 s silence timeout.
        """
        assert self._proc is not None
        assert self._proc.stderr is not None

        for raw_line in self._proc.stderr:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                print(f"[piper] {line}")

            # End-of-utterance signal: piper finished writing PCM for this line
            if "Real-time factor" in line:
                m = re.search(r"infer=([0-9.]+)\s*sec,\s*audio=([0-9.]+)\s*sec", line)
                if m:
                    try:
                        self._last_infer_sec = float(m.group(1))
                        self._last_audio_sec = float(m.group(2))
                    except ValueError:
                        self._last_infer_sec = None
                        self._last_audio_sec = None
                self._utterance_done.set()

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Spawn piper and wait for it to be ready.
        Returns True on success, False if startup fails.
        """
        # Validate paths before launching
        if not PIPER_EXE.exists():
            print(
                f"[TTS] ✗ Piper exe not found: {PIPER_EXE}\n"
                "       → Extract the full Piper release zip into models/piper/"
            )
            return False
        if not PIPER_MODEL_PATH.exists():
            print(f"[TTS] ✗ Piper model not found: {PIPER_MODEL_PATH}")
            return False

        cmd = self._build_cmd()
        print(f"[TTS] Starting persistent Piper process…")
        print(f"[TTS]   exe  : {PIPER_EXE}")
        print(f"[TTS]   model: {PIPER_MODEL_PATH}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Ensure piper doesn't buffer stdout (critical for streaming)
                bufsize=0,
            )
        except FileNotFoundError as exc:
            print(f"[TTS] ✗ Could not launch piper: {exc}")
            return False
        except PermissionError as exc:
            print(f"[TTS] ✗ Permission denied launching piper: {exc}")
            return False

        # Give piper a moment to load the model before first use
        # On Jetson this can take up to ~900 ms — we probe with a silent line
        time.sleep(0.1)
        rc = self._proc.poll()
        if rc is not None:
            self._diagnose_exit(rc)
            return False

        # Start background threads
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="piper-reader"
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_monitor, daemon=True, name="piper-stderr"
        )
        self._reader_thread.start()
        self._stderr_thread.start()

        # Warm-up: send a silent test utterance so the model is fully loaded
        # before the first real speak() call. This moves the load latency to
        # startup rather than the first user-facing response.
        print("[TTS] Warming up Piper (loading ONNX model)…")
        self._warmup()

        print("[TTS] ✓ Piper ready — persistent process active.")
        self._ok = True
        return True

    def _warmup(self) -> None:
        """
        Send a short phrase to force model load, discard audio output.
        Uses _utterance_done to know exactly when piper finishes — no fixed sleep.
        """
        assert self._proc is not None
        assert self._proc.stdin is not None
        try:
            self._utterance_done.clear()
            self._utterance_active.set()
            self._proc.stdin.write(b" \n")
            self._proc.stdin.flush()
            # Wait for piper to signal completion via stderr (model load + synth)
            # Generous timeout for Jetson cold-start (model load can take ~2 s)
            self._utterance_done.wait(timeout=5.0)
        except OSError:
            pass
        finally:
            self._utterance_active.clear()
            self._utterance_done.clear()
            # Flush any warmup audio from the queue
            while not self._audio_q.empty():
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break

    def _diagnose_exit(self, rc: int) -> None:
        if rc == _WIN_DLL_MISSING or rc == -1073741515:
            print(
                "[TTS] ✗ Piper exited with 0xC0000135 — missing DLL.\n"
                "       Extract ALL files from piper_windows_amd64.zip into models/piper/"
            )
        else:
            print(f"[TTS] ✗ Piper exited unexpectedly (code {rc}).")

    def speak(self, text: str) -> dict[str, float]:
        """
        Send text to the persistent piper process and stream PCM to speakers.
        Blocks until playback is complete.

        End-of-utterance detection
        ──────────────────────────
        Previously used a silence-streak timeout (60 × 20 ms = 1.2 s guaranteed
        wait after the last audio chunk). This caused 5–10 s TTS latency.

        Now: _stderr_monitor sets _utterance_done the moment piper writes the
        "Real-time factor:" line to stderr, which happens immediately after the
        last PCM byte is flushed to stdout. We drain the audio queue until that
        event fires, then play any remaining buffered chunks and return.
        This eliminates the 1.2 s tail wait entirely.
        """
        import sounddevice as sd

        with self._lock:
            start = time.perf_counter()
            assert self._proc is not None
            assert self._proc.stdin is not None

            # Check process is still alive
            if self._proc.poll() is not None:
                raise RuntimeError("Piper process died unexpectedly.")

            # Clear any stale queue data and reset the done event
            while not self._audio_q.empty():
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break
            self._utterance_done.clear()
            self._last_infer_sec = None
            self._last_audio_sec = None

            # Signal reader thread to start enqueuing audio
            self._utterance_active.set()

            # Send text to piper (one line = one utterance)
            line = (text.strip() + "\n").encode("utf-8")
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

            # ── Event-driven streaming playback ─────────────────────────────
            # Open the audio stream before reading the queue so the output
            # device is ready when the first chunk arrives.
            with sd.RawOutputStream(
                samplerate=PIPER_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=STREAM_CHUNK_BYTES // 2,   # frames (2 bytes each)
            ) as stream:

                # Phase 1: stream audio until piper signals "done" via stderr
                while not self._utterance_done.is_set():
                    try:
                        chunk = self._audio_q.get(timeout=0.01)
                    except queue.Empty:
                        continue

                    if chunk is None:          # piper process died
                        self._ok = False
                        self._utterance_active.clear()
                        total = time.perf_counter() - start
                        return {"inference": total, "speaking": 0.0, "total": total}

                    stream.write(chunk)

                # Phase 2: piper is done generating — drain any remaining
                # buffered chunks that arrived before we saw the event.
                while True:
                    try:
                        chunk = self._audio_q.get_nowait()
                        if chunk is None:
                            break
                        stream.write(chunk)
                    except queue.Empty:
                        break

            self._utterance_active.clear()

            total = time.perf_counter() - start
            infer = self._last_infer_sec if self._last_infer_sec is not None else max(0.0, total * 0.25)
            speaking = self._last_audio_sec if self._last_audio_sec is not None else max(0.0, total - infer)
            return {"inference": infer, "speaking": speaking, "total": total}

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None
        self._ok   = False
        print("[TTS] Piper process shut down.")


# ── Module-level singleton ──────────────────────────────────────────────────
_piper: _PiperProcess | None = None
_piper_ok: bool | None = None   # None=unchecked, True=ok, False=failed


def _ensure_piper() -> bool:
    """Start the persistent Piper process if not already running."""
    global _piper, _piper_ok

    if _piper_ok is True and _piper is not None and _piper.is_alive():
        return True

    if _piper_ok is False:
        return False  # already diagnosed as broken — don't retry

    _piper = _PiperProcess()
    _piper_ok = _piper.start()

    if _piper_ok:
        atexit.register(_piper.shutdown)

    return _piper_ok


# ══════════════════════════════════════════════════════════════════════════════
#  OPENAI STREAMING
# ══════════════════════════════════════════════════════════════════════════════

def _build_openai_client():
    try:
        openai  = importlib.import_module("openai")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        return openai.OpenAI(api_key=api_key)
    except ImportError:
        return None


_openai_client: object | None = None
_openai_ready: bool | None    = None


def _get_openai_client():
    global _openai_client, _openai_ready
    if _openai_ready is None:
        _openai_client = _build_openai_client()
        _openai_ready  = _openai_client is not None
    return _openai_client if _openai_ready else None


def _speak_openai(text: str, client) -> dict[str, float]:
    import sounddevice as sd

    audio_q: queue.Queue[bytes | None] = queue.Queue(maxsize=64)
    timing = {
        "start": time.perf_counter(),
        "first_chunk": None,
    }

    def _fetch():
        try:
            with client.audio.speech.with_streaming_response.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=text,
                response_format=OPENAI_TTS_FORMAT,
                speed=OPENAI_TTS_SPEED,
            ) as resp:
                for chunk in resp.iter_bytes(chunk_size=STREAM_CHUNK_BYTES):
                    if chunk:
                        if timing["first_chunk"] is None:
                            timing["first_chunk"] = time.perf_counter()
                        audio_q.put(chunk)
        except Exception as exc:
            print(f"[TTS] OpenAI stream error: {exc}")
        finally:
            audio_q.put(None)

    threading.Thread(target=_fetch, daemon=True).start()

    with sd.RawOutputStream(samplerate=OPENAI_TTS_RATE, channels=1, dtype="int16") as stream:
        while True:
            chunk = audio_q.get()
            if chunk is None:
                break
            stream.write(chunk)

    end = time.perf_counter()
    start = timing["start"]
    first = timing["first_chunk"]
    total = end - start
    if first is None:
        return {"inference": total, "speaking": 0.0, "total": total}
    return {
        "inference": max(0.0, first - start),
        "speaking": max(0.0, end - first),
        "total": total,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PYTTSX3 FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _speak_pyttsx3(text: str) -> dict[str, float]:
    """Re-init pyttsx3 each call to avoid the silent-after-first-call bug."""
    start = time.perf_counter()
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate",   PYTTSX3_RATE)
    engine.setProperty("volume", PYTTSX3_VOLUME)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    total = time.perf_counter() - start
    return {"inference": 0.0, "speaking": total, "total": total}


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

_provider_logged = False


def _tts_provider() -> str:
    p = os.getenv("TTS_PROVIDER", "piper").strip().lower()
    if p not in {"piper", "openai", "pyttsx3"}:
        print(f"[TTS] Unknown TTS_PROVIDER='{p}', defaulting to 'piper'.")
        return "piper"
    return p


def speak(text: str) -> None:
    """
    Convert text to speech and block until playback is complete.

    Piper path:  persistent process + real-time streaming (lowest latency).
    OpenAI path: streaming PCM over HTTPS.
    pyttsx3:     fallback (blocks, no streaming).
    """
    global _provider_logged

    if not text or not text.strip():
        return

    provider = _tts_provider()

    if not _provider_logged:
        print(f"[TTS] Provider: {provider}")
        _provider_logged = True

    print(f"[TTS] {text}")

    speak_with_timings(text)


def speak_with_timings(text: str) -> dict[str, float]:
    """Speak text and return timing split: inference, speaking, total."""
    global _provider_logged

    if not text or not text.strip():
        return {"inference": 0.0, "speaking": 0.0, "total": 0.0}

    provider = _tts_provider()

    if not _provider_logged:
        print(f"[TTS] Provider: {provider}")
        _provider_logged = True

    print(f"[TTS] {text}")

    try:
        if provider == "piper":
            if not _ensure_piper():
                print("[TTS] Piper unavailable — using pyttsx3.")
                return _speak_pyttsx3(text)
            assert _piper is not None
            return _piper.speak(text)

        if provider == "openai":
            client = _get_openai_client()
            if not client:
                raise RuntimeError("OpenAI unavailable — check OPENAI_API_KEY.")
            return _speak_openai(text, client)

        return _speak_pyttsx3(text)

    except Exception as exc:
        print(f"[TTS] Provider '{provider}' failed: {exc}")
        print("[TTS] Falling back to pyttsx3.")
        return _speak_pyttsx3(text)


def warmup_tts() -> None:
    """
    Call this at application startup to pre-load Piper's ONNX model.
    Eliminates first-utterance latency entirely.
    The persistent process stays alive for the entire session.
    """
    provider = _tts_provider()
    if provider == "piper":
        _ensure_piper()   # spawns process + warms model