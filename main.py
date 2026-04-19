# main.py — Phase 4 + Performance Fixes
#
# FIX 2 — Non-blocking TTS:
#   speak() is now launched in a daemon thread so the system remains
#   responsive while audio plays. A threading.Event signals completion
#   before the next recording starts (we still need to finish speaking
#   before listening again, but other work can proceed in parallel).
#
# FIX 3 — Reduced speech length:
#   LLM instructions now explicitly request ultra-short replies so TTS
#   time drops drastically (fewer words = fewer seconds of audio).

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import threading

import numpy as np
import sounddevice as sd
from time import perf_counter

from audio.recorder import record_audio, SAMPLE_RATE
from stt.stt import transcribe, _load_model
from llm.factory import get_llm_client
from llm.base import BaseLLMClient
from state.session import SessionManager
from tts.tts import speak

# ── Silence retry config ───────────────────────────────────────────────────
SILENCE_RETRY_AFTER  = 1
SILENCE_RETRY_PROMPT = "Didn't catch that — please speak clearly."
BEEP_FREQ_HZ         = 880
BEEP_DURATION_SECS   = 0.18
BEEP_VOLUME          = 0.4
# ──────────────────────────────────────────────────────────────────────────


def _play_beep() -> None:
    """Play a short sine-wave attention beep."""
    t    = np.linspace(0, BEEP_DURATION_SECS,
                       int(SAMPLE_RATE * BEEP_DURATION_SECS), endpoint=False)
    wave = BEEP_VOLUME * np.sin(2 * np.pi * BEEP_FREQ_HZ * t)
    fade = np.linspace(1.0, 0.0, len(wave) // 4)
    wave[-len(fade):] *= fade
    sd.play(wave.astype(np.float32), samplerate=SAMPLE_RATE)
    sd.wait()


# ── FIX 2: Non-blocking speak ───────────────────────────────────────────────

def speak_async(text: str) -> threading.Event:
    """
    Start TTS in a background thread; return an Event that is set when done.
    Callers can do other work and call event.wait() only when they need
    playback to have finished (e.g., before starting the next recording).
    """
    done = threading.Event()

    def _run():
        speak(text)
        done.set()

    threading.Thread(target=_run, daemon=True).start()
    return done


def get_assistant_response(llm: BaseLLMClient, instruction: str) -> str:
    return llm.generate_response(instruction)


def run_turn(llm: BaseLLMClient, session: SessionManager) -> tuple[str, str, dict[str, float]]:
    """
    One full conversation turn.
    Returns (user_text, assistant_text, metrics).
    Returns ('', '', metrics) when no speech was captured.
    """
    metrics = {
        "user_wait": 0.0,
        "stt":       0.0,
        "llm":       0.0,
        "tts":       0.0,
        "processing_total": 0.0,
        "total":     0.0,
    }

    turn_start   = perf_counter()
    active_slot  = session.current_slot()

    listen_start = perf_counter()
    if active_slot == "name":
        audio = record_audio(silence_duration=1.8, min_speech_frames=5,
                              pre_roll_chunks=6, vad_threshold=260)
    elif active_slot == "phone":
        audio = record_audio(silence_duration=2.4, min_speech_frames=4,
                              pre_roll_chunks=8, vad_threshold=220)
    else:
        audio = record_audio()
    metrics["user_wait"] = perf_counter() - listen_start

    stt_start    = perf_counter()
    user_text    = transcribe(audio, hint=active_slot)
    metrics["stt"] = perf_counter() - stt_start

    if not user_text:
        metrics["processing_total"] = metrics["stt"]
        metrics["total"]            = perf_counter() - turn_start
        return "", "", metrics

    session.update(user_text)

    instruction    = session.get_next_prompt()
    llm_start      = perf_counter()
    assistant_text = get_assistant_response(llm, instruction)
    metrics["llm"] = perf_counter() - llm_start

    # FIX 2 — speak in background; wait for it to finish before returning
    tts_start  = perf_counter()
    tts_done   = speak_async(assistant_text)
    tts_done.wait()                          # blocks until audio ends
    metrics["tts"] = perf_counter() - tts_start

    metrics["processing_total"] = metrics["stt"] + metrics["llm"] + metrics["tts"]
    metrics["total"]            = perf_counter() - turn_start
    return user_text, assistant_text, metrics


def _print_latency(metrics: dict[str, float]) -> None:
    print("[Latency] User wait: {:.1f}s".format(metrics["user_wait"]))
    print("[Latency] STT:       {:.1f}s".format(metrics["stt"]))
    print("[Latency] LLM:       {:.1f}s".format(metrics["llm"]))
    print("[Latency] TTS:       {:.1f}s".format(metrics["tts"]))
    print("[Latency] Processing:{:.1f}s".format(metrics["processing_total"]))
    print("[Latency] Total:      {:.1f}s".format(metrics["total"]))


def run_session(llm: BaseLLMClient) -> None:
    """Run a single user session."""
    session = SessionManager()

    opening_instruction = session.get_next_prompt()
    opening_message     = get_assistant_response(llm, opening_instruction)
    print(f"\n[Assistant] {opening_message}")

    # FIX 2: opening message is also non-blocking — beep fires sooner
    tts_done = speak_async(opening_message)
    tts_done.wait()
    _play_beep()

    silent_turns = 0

    while not session.is_complete() and not session.is_failed():
        user_text, response, metrics = run_turn(llm, session)

        if not user_text:
            silent_turns += 1
            print(f"[main] No speech captured (silent turn #{silent_turns}).")
            _print_latency(metrics)

            if silent_turns >= SILENCE_RETRY_AFTER:
                tts_done = speak_async(SILENCE_RETRY_PROMPT)
                tts_done.wait()
                _play_beep()
                silent_turns = 0
            continue

        silent_turns = 0
        print(f"[You]       {user_text}")
        print(f"[Assistant] {response}")
        _print_latency(metrics)

        if not session.is_complete() and not session.is_failed():
            _play_beep()

    if session.is_complete():
        print("\n[main] Session complete.")
        print(session.summary())
    else:
        print("\n[main] Session ended — retry limit reached.")
        print(session.summary())


def main():
    print("=== Voice Assistant — Phase 4 (optimized) ===")
    print("Press Ctrl+C to quit.\n")

    print("[main] Pre-loading Whisper STT model...")
    _load_model()
    print("[main] STT model ready.\n")

    llm = get_llm_client()

    while True:
        try:
            print("\n--- New user session ---")
            run_session(llm)
        except KeyboardInterrupt:
            print("\n[main] Exiting.")
            break


if __name__ == "__main__":
    main()