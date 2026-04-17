# stt.py — Speech-to-Text using local OpenAI Whisper
# Accepts a numpy audio array, returns transcribed text.
#
# FIXES:
#   1. no_speech_prob check: Whisper returns a confidence score that the audio
#      contains no speech. If above NO_SPEECH_THRESHOLD, return "" instead of
#      hallucinated text (e.g. "Thank you." "I'm sorry." that Whisper invents
#      for silence/noise).
#   2. Minimum audio duration guard (was only checking size == 0).
#   3. Log no_speech_prob so you can tune the threshold.

import numpy as np
import whisper

_model = None          # lazy-loaded so import is fast
MODEL_SIZE = "base"    # tiny | base | small | medium | large

# Whisper's no_speech_prob threshold — if the model is this confident the audio
# has no speech, discard it. Range 0.0–1.0. Lower = stricter.
# Raise this if real speech is being discarded; lower if noise leaks through.
NO_SPEECH_THRESHOLD = 0.6

# Minimum audio length in seconds to even attempt transcription
MIN_AUDIO_SECS = 0.3


def _load_model():
    global _model
    if _model is None:
        print(f"[stt] Loading Whisper model '{MODEL_SIZE}'...")
        _model = whisper.load_model(MODEL_SIZE)
    return _model


def _initial_prompt_for_hint(hint: str | None) -> str | None:
    """Return Whisper initial prompt guidance for slot-specific decoding."""
    if hint == "phone":
        return (
            "The user is speaking a 10 digit phone number. "
            "Transcribe digits clearly as numbers."
        )
    if hint == "name":
        return (
            "The user is stating their name. "
            "Transcribe only the spoken name words."
        )
    return None


def transcribe(audio: np.ndarray, sample_rate: int = 16000, hint: str | None = None) -> str:
    """
    Convert a 1-D int16 numpy array to text.
    Returns an empty string if audio is too short or Whisper detects no speech.
    """
    if audio.size == 0:
        return ""

    # Guard: skip clips shorter than MIN_AUDIO_SECS
    duration = audio.size / sample_rate
    if duration < MIN_AUDIO_SECS:
        print(f"[stt] Audio too short ({duration:.2f}s) — skipping.")
        return ""

    model = _load_model()

    # Whisper expects float32 in [-1, 1]
    audio_f32 = audio.astype(np.float32) / 32768.0

    result = model.transcribe(
        audio_f32,
        fp16=False,
        language="en",
        initial_prompt=_initial_prompt_for_hint(hint),
        # Return per-segment details so we can check no_speech_prob
        verbose=False,
    )

    # Check overall no-speech confidence
    # result["segments"] is a list; each has no_speech_prob
    segments = result.get("segments", [])
    if segments:
        avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
        print(f"[stt] no_speech_prob={avg_no_speech:.2f}  duration={duration:.1f}s")
        if avg_no_speech > NO_SPEECH_THRESHOLD:
            print(f"[stt] Discarded — likely noise (no_speech_prob {avg_no_speech:.2f} > {NO_SPEECH_THRESHOLD})")
            return ""
    else:
        print(f"[stt] No segments returned (duration={duration:.1f}s).")
        return ""

    text = result.get("text", "").strip()
    return text