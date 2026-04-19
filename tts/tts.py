"""TTS provider layer.

Default provider is Piper (local/offline) and can be changed via .env:
    TTS_PROVIDER=piper | openai | pyttsx3

Callers keep using speak(text) with no API changes.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import importlib

# ── Config ─────────────────────────────────────────────────────────────────
STREAM_CHUNK_BYTES = 4096

OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
OPENAI_TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.0"))
OPENAI_TTS_FORMAT = "pcm"
OPENAI_TTS_RATE = int(os.getenv("OPENAI_TTS_RATE", "24000"))

PIPER_EXE = os.getenv("PIPER_EXE", "piper")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "").strip()
PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", "").strip()
PIPER_SAMPLE_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))

PYTTSX3_RATE   = 170
PYTTSX3_VOLUME = 1.0
# ───────────────────────────────────────────────────────────────────────────


def _build_openai_client():
    """Return an openai.OpenAI client, or None if unavailable/no key."""
    try:
        openai = importlib.import_module("openai")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        return openai.OpenAI(api_key=api_key)
    except ImportError:
        return None


# Lazy-init once per process
_openai_client = None
_openai_ready: bool | None = None   # None = not yet probed
_provider_logged = False


def _get_openai_client():
    global _openai_client, _openai_ready
    if _openai_ready is None:
        _openai_client = _build_openai_client()
        _openai_ready  = _openai_client is not None
        tag = "OpenAI TTS (streaming PCM)" if _openai_ready else "pyttsx3 (fallback)"
        print(f"[TTS] Engine: {tag}")
    return _openai_client if _openai_ready else None


# ── OpenAI streaming speak ─────────────────────────────────────────────────

def _speak_openai(text: str, client) -> None:
    """
    Stream PCM from OpenAI TTS and play in real time via sounddevice.
    """
    import sounddevice as sd

    audio_q: queue.Queue[bytes | None] = queue.Queue(maxsize=32)

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
                        audio_q.put(chunk)
        except Exception as exc:
            print(f"[TTS] OpenAI stream error: {exc}")
        finally:
            audio_q.put(None)   # sentinel

    fetcher = threading.Thread(target=_fetch, daemon=True)
    fetcher.start()

    with sd.RawOutputStream(samplerate=OPENAI_TTS_RATE, channels=1, dtype="int16") as stream:
        while True:
            chunk = audio_q.get()
            if chunk is None:
                break
            stream.write(chunk)

    fetcher.join(timeout=5)


def _speak_piper(text: str) -> None:
    """Run Piper CLI and stream raw PCM output to sounddevice."""
    if not PIPER_MODEL_PATH:
        raise RuntimeError("PIPER_MODEL_PATH is not set. Configure it in .env")

    import sounddevice as sd

    cmd = [PIPER_EXE, "--model", PIPER_MODEL_PATH, "--output-raw"]
    if PIPER_CONFIG_PATH:
        cmd.extend(["--config", PIPER_CONFIG_PATH])

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Piper executable not found: {PIPER_EXE}") from exc

    assert proc.stdin is not None
    assert proc.stdout is not None

    proc.stdin.write((text.strip() + "\n").encode("utf-8"))
    proc.stdin.close()

    with sd.RawOutputStream(samplerate=PIPER_SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while True:
            chunk = proc.stdout.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            stream.write(chunk)

    return_code = proc.wait(timeout=10)
    if return_code != 0:
        stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"Piper TTS failed (exit {return_code}): {stderr.strip()}")


# ── Fallback: pyttsx3 ───────────────────────────────────────────────────────

def _speak_pyttsx3(text: str) -> None:
    """Re-init pyttsx3 each call to avoid the silent-after-first-call bug."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate",   PYTTSX3_RATE)
    engine.setProperty("volume", PYTTSX3_VOLUME)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def _tts_provider() -> str:
    provider = os.getenv("TTS_PROVIDER", "piper").strip().lower()
    if provider not in {"piper", "openai", "pyttsx3"}:
        print(f"[TTS] Unknown TTS_PROVIDER='{provider}', falling back to 'piper'.")
        return "piper"
    return provider


# ── Public API ───────────────────────────────────────────────────────────────

def speak(text: str) -> None:
    """
    Convert text to speech and block until playback is complete.
    Primary provider: Piper. OpenAI selectable via .env.
    """
    global _provider_logged
    if not text or not text.strip():
        return

    provider = _tts_provider()
    if not _provider_logged:
        print(f"[TTS] Provider: {provider}")
        _provider_logged = True

    print(f"[TTS] {text}")
    try:
        if provider == "piper":
            _speak_piper(text)
            return
        if provider == "openai":
            client = _get_openai_client()
            if not client:
                raise RuntimeError("OpenAI client unavailable. Check OPENAI_API_KEY and openai package.")
            _speak_openai(text, client)
            return
        _speak_pyttsx3(text)
    except Exception as exc:
        print(f"[TTS] Provider '{provider}' failed: {exc}")
        print("[TTS] Falling back to pyttsx3.")
        _speak_pyttsx3(text)