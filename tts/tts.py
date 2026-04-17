# tts/tts.py — Phase 4: Text-to-Speech output
# Uses pyttsx3 (fully local, no API key, cross-platform).
# Phase 5+ can swap speak() internals for ElevenLabs / gTTS without touching callers.
#
# FIX: pyttsx3's runAndWait() silently fails on subsequent calls when the engine
# is reused (the internal event loop goes stale after the first completion).
# Solution: reinitialise the engine on every speak() call — cheap and reliable.

import pyttsx3

_RATE   = 160   # words per minute
_VOLUME = 1.0   # 0.0 – 1.0


def speak(text: str) -> None:
    """
    Convert text to speech and block until playback is complete.
    A fresh pyttsx3 engine is created each call to avoid the silent-after-first
    runAndWait() bug that appears when the engine instance is reused across turns.
    """
    if not text or not text.strip():
        return

    print(f"[TTS] {text}")   # console mirror — useful for debugging

    engine = pyttsx3.init()
    engine.setProperty("rate",   _RATE)
    engine.setProperty("volume", _VOLUME)
    engine.say(text)
    engine.runAndWait()
    engine.stop()            # release driver resources cleanly before next init