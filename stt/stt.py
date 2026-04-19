# stt.py — Speech-to-Text using local OpenAI Whisper
#
# FIXES:
#   1. Removed initial_prompt — Whisper 'base' bleeds the prompt text directly
#      into the transcription output (seen as "The user is speaking a 10 digit
#      phone number." appended verbatim). Use condition_on_previous_text=False
#      and slot-aware post-processing instead of prompting.
#   2. condition_on_previous_text=False — prevents the model from conditioning
#      on hallucinated prior context across segments, which compounds errors.
#   3. Slot-aware post-processing (_clean_output) strips any residual noise
#      tokens from transcription without touching actual speech content.

import re
import numpy as np
import whisper

_model = None
MODEL_SIZE = "base"   # tiny | base | small | medium | large

NO_SPEECH_THRESHOLD = 0.6   # discard if avg no_speech_prob above this
MIN_AUDIO_SECS      = 0.3   # skip clips shorter than this


def _load_model():
    global _model
    if _model is None:
        print(f"[stt] Loading Whisper model '{MODEL_SIZE}'...")
        _model = whisper.load_model(MODEL_SIZE)
    return _model


# ── Slot-aware post-processing ───────────────────────────────────────────────

def _clean_phone(text: str) -> str:
    """
    Keep only digit-like tokens from a phone transcription.
    Strips any sentence fragments Whisper may emit alongside numbers.
    """
    # Extract all digit sequences and number words; discard prose
    NUMBER_WORDS = {
        "zero", "oh", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "double", "triple",
    }
    tokens = re.findall(r"[a-zA-Z]+|\d+", text.lower())
    kept = []
    for tok in tokens:
        if tok.isdigit() or tok in NUMBER_WORDS:
            kept.append(tok)
    return " ".join(kept) if kept else text


def _clean_name(text: str) -> str:
    """Strip trailing punctuation / filler from a name transcription."""
    # Remove anything after a sentence-ending punctuation followed by more words
    # e.g. "Abraham. The user is ..." → "Abraham"
    text = re.split(r"[.!?]\s+[A-Z]", text)[0]
    return text.strip().rstrip(".,!?")


_CLEANERS = {
    "phone": _clean_phone,
    "name":  _clean_name,
}


# ── Main transcribe function ─────────────────────────────────────────────────

def transcribe(audio: np.ndarray, sample_rate: int = 16000,
               hint: str | None = None) -> str:
    """
    Convert a 1-D int16 numpy array to text.
    Returns empty string if audio is silent, too short, or transcription fails.

    `hint` — slot name ('name', 'phone') used for post-processing only.
    initial_prompt is intentionally NOT used: Whisper base bleeds prompt
    text verbatim into the output, corrupting transcriptions.
    """
    if audio.size == 0:
        return ""

    duration = audio.size / sample_rate
    if duration < MIN_AUDIO_SECS:
        print(f"[stt] Audio too short ({duration:.2f}s) — skipping.")
        return ""

    model = _load_model()
    audio_f32 = audio.astype(np.float32) / 32768.0

    result = model.transcribe(
        audio_f32,
        fp16=False,
        language="en",
        # NO initial_prompt — it bleeds into output on 'base' model
        condition_on_previous_text=False,   # prevents compounding hallucinations
        verbose=False,
    )

    segments = result.get("segments", [])
    if not segments:
        print(f"[stt] No segments (duration={duration:.1f}s).")
        return ""

    avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
    print(f"[stt] no_speech_prob={avg_no_speech:.2f}  duration={duration:.1f}s")

    if avg_no_speech > NO_SPEECH_THRESHOLD:
        print(f"[stt] Discarded — likely noise (no_speech_prob={avg_no_speech:.2f})")
        return ""

    text = result.get("text", "").strip()

    # Apply slot-aware cleaner to strip any residual hallucinated prose
    cleaner = _CLEANERS.get(hint or "")
    if cleaner and text:
        cleaned = cleaner(text)
        if cleaned != text:
            print(f"[stt] Cleaned ({hint}): '{text}' → '{cleaned}'")
        text = cleaned

    return text