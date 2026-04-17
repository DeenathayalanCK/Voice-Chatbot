# vad.py — Voice Activity Detection
# Determines whether an audio chunk contains speech using energy threshold.
# Designed to be swappable with a model-based VAD in future phases.
#
# FIX: The original threshold of 500 was too LOW — it triggered on ambient mic
# hiss/noise, so the recorder thought speech was happening when it wasn't, and
# passed short garbage audio to Whisper which returned "".
#
# HOW TO CALIBRATE FOR YOUR MIC:
#   Run this file directly:  python -m audio.vad
#   Sit quietly for 3 seconds — it will print your ambient RMS.
#   Set SILENCE_THRESHOLD to roughly 3-5x that value.
#   Typical USB mic in a quiet room: ambient ~30-80 RMS  → threshold ~200-400
#   Laptop mic in a noisy room:      ambient ~200-600 RMS → threshold ~1000-2000

import numpy as np

# ── Tune this to your mic ──────────────────────────────────────────────────
# Too LOW  → fires on noise, Whisper gets garbage audio, STT returns ""
# Too HIGH → misses quiet speech
SILENCE_THRESHOLD = 300    # RMS energy below this = silence (was 500, which is paradoxically too sensitive)

# Seconds of sustained silence before recording stops
SILENCE_DURATION  = 1.5    # slightly tighter than 2 s for snappier UX

# Minimum speech frames required before we consider a recording valid.
# Prevents sub-100 ms noise bursts from being sent to Whisper.
MIN_SPEECH_FRAMES = 8      # x 32 ms/chunk = ~256 ms minimum utterance
# ──────────────────────────────────────────────────────────────────────────


def detect_voice_activity(audio_chunk: np.ndarray, threshold: float | None = None) -> bool:
    """Return True if the chunk contains speech (energy above threshold)."""
    rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
    active_threshold = SILENCE_THRESHOLD if threshold is None else threshold
    return rms > active_threshold


def is_silence(audio_chunk: np.ndarray) -> bool:
    """Convenience inverse — True when no speech detected."""
    return not detect_voice_activity(audio_chunk)


# ── Self-calibration helper ────────────────────────────────────────────────
if __name__ == "__main__":
    import sounddevice as sd

    SAMPLE_RATE  = 16000
    CHUNK_FRAMES = 512
    CALIBRATE_SECS = 3

    print(f"[calibrate] Measuring ambient noise for {CALIBRATE_SECS} s — stay quiet...")
    chunks = int(CALIBRATE_SECS * SAMPLE_RATE / CHUNK_FRAMES)
    rms_values = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=CHUNK_FRAMES) as stream:
        for _ in range(chunks):
            chunk, _ = stream.read(CHUNK_FRAMES)
            chunk = chunk[:, 0]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            rms_values.append(rms)

    ambient = float(np.mean(rms_values))
    peak    = float(np.max(rms_values))
    print(f"\n  Ambient RMS (mean): {ambient:.1f}")
    print(f"  Ambient RMS (peak): {peak:.1f}")
    print(f"\n  Recommended SILENCE_THRESHOLD: {int(peak * 4)}")
    print(f"  (Current value: {SILENCE_THRESHOLD})")
    print("\nNow speak normally for 6 seconds — watch the RMS values:")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=CHUNK_FRAMES) as stream:
        for _ in range(chunks * 2):
            chunk, _ = stream.read(CHUNK_FRAMES)
            chunk = chunk[:, 0]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            bar = "█" * int(rms / 50)
            print(f"  RMS {rms:6.1f}  {bar}")