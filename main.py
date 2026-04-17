# main.py — Phase 4: full POC with TTS, validation, retry logic, multi-user loop
#
# FIXES in this revision:
#   1. pyttsx3 silent-after-first-call → fixed in tts.py (reinit per call)
#   2. ACK + next question chained → fixed in session.get_next_prompt()
#   3. Silence retry: speaks prompt + plays an actual BEEP tone (no more
#      "after the beep" with no beep)
#   4. Whisper pre-warmed at startup
#   5. No-speech detection improved in recorder.py + stt.py

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import sounddevice as sd

from audio.recorder import record_audio, SAMPLE_RATE
from stt.stt import transcribe, _load_model
from llm.factory import get_llm_client
from llm.base import BaseLLMClient
from state.session import SessionManager
from tts.tts import speak

# ── Silence retry config ───────────────────────────────────────────────────
SILENCE_RETRY_AFTER   = 1      # speak retry prompt after this many silent turns
SILENCE_RETRY_PROMPT  = "I didn't catch that — please speak clearly."
BEEP_FREQ_HZ          = 880    # A5 — pleasant attention tone
BEEP_DURATION_SECS    = 0.18   # short enough not to be annoying
BEEP_VOLUME           = 0.4    # 0.0–1.0
# ──────────────────────────────────────────────────────────────────────────


def _play_beep() -> None:
    """Play a short sine-wave beep through the default output device."""
    t = np.linspace(0, BEEP_DURATION_SECS,
                    int(SAMPLE_RATE * BEEP_DURATION_SECS), endpoint=False)
    # Sine wave with a short fade-out to avoid click at the end
    wave = BEEP_VOLUME * np.sin(2 * np.pi * BEEP_FREQ_HZ * t)
    fade = np.linspace(1.0, 0.0, len(wave) // 4)
    wave[-len(fade):] *= fade
    sd.play(wave.astype(np.float32), samplerate=SAMPLE_RATE)
    sd.wait()


def get_assistant_response(llm: BaseLLMClient, instruction: str) -> str:
    """Ask the LLM to phrase a response based on a flow instruction."""
    return llm.generate_response(instruction)


def run_turn(llm: BaseLLMClient, session: SessionManager) -> tuple[str, str]:
    """
    One full conversation turn:
      1. Capture voice → transcribe
      2. Validate + update session state
      3. Get LLM-phrased response
      4. Speak response aloud
    Returns (user_text, assistant_text).
    Returns ("", "") when no speech was captured.
    """
    active_slot = session.current_slot()

    # Name capture can be short; relax minimum voiced frames slightly.
    if active_slot == "name":
        audio = record_audio(
            silence_duration=1.8,
            min_speech_frames=5,
            pre_roll_chunks=6,
            vad_threshold=260,
        )
    # Phone capture needs more forgiving timing because users pause between digits.
    # These values reduce clipping of first/last digits and avoid early stop mid-number.
    elif active_slot == "phone":
        audio = record_audio(
            silence_duration=2.4,
            min_speech_frames=4,
            pre_roll_chunks=8,
            vad_threshold=220,
        )
    else:
        audio = record_audio()

    user_text   = transcribe(audio, hint=active_slot)

    if not user_text:
        return "", ""

    session.update(user_text)

    instruction    = session.get_next_prompt()
    assistant_text = get_assistant_response(llm, instruction)

    speak(assistant_text)
    return user_text, assistant_text


def run_session(llm: BaseLLMClient) -> None:
    """Run a single user session. Exits when all slots filled or retries exhausted."""
    session = SessionManager()

    # Opening prompt — speak before first recording
    opening_instruction = session.get_next_prompt()
    opening_message     = get_assistant_response(llm, opening_instruction)
    print(f"\n[Assistant] {opening_message}")
    speak(opening_message)
    _play_beep()   # signal: ready to listen

    silent_turns = 0

    while not session.is_complete() and not session.is_failed():
        user_text, response = run_turn(llm, session)

        if not user_text:
            silent_turns += 1
            print(f"[main] No speech captured (silent turn #{silent_turns}).")

            if silent_turns >= SILENCE_RETRY_AFTER:
                speak(SILENCE_RETRY_PROMPT)
                _play_beep()      # beep AFTER the spoken prompt, as promised
                silent_turns = 0
            continue

        silent_turns = 0
        print(f"[You]       {user_text}")
        print(f"[Assistant] {response}")

        # After a valid response, beep to signal we're ready for the next input
        if not session.is_complete() and not session.is_failed():
            _play_beep()

    # Final outcome
    if session.is_complete():
        print("\n[main] Session complete.")
        print(session.summary())
    else:
        print("\n[main] Session ended — retry limit reached.")
        print(session.summary())


def main():
    print("=== Voice Assistant — Phase 4 (multi-provider) ===")
    print("Press Ctrl+C to quit.\n")

    # Pre-warm Whisper so first turn has no STT delay
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