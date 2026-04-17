# recorder.py — Microphone capture with VAD-gated recording
# Listens continuously, starts buffering on speech, stops on sustained silence.
# Returns a raw numpy array ready for STT.
#
# FIXES:
#   1. PRE-ROLL BUFFER: Keep the last N chunks before speech trigger so we don't
#      clip the very first syllable (was causing Whisper to miss short words).
#   2. MIN_SPEECH_FRAMES guard: If fewer than MIN_SPEECH_FRAMES voiced frames were
#      captured, discard and return empty — prevents noise bursts from reaching STT.
#   3. RMS logging: Print RMS on speech detection to help calibrate VAD threshold.

import collections
import numpy as np
import sounddevice as sd
from audio.vad import detect_voice_activity, SILENCE_DURATION, MIN_SPEECH_FRAMES

SAMPLE_RATE   = 16000   # Hz — Whisper expects 16 kHz
CHANNELS      = 1
CHUNK_FRAMES  = 512     # frames per VAD evaluation (~32 ms at 16 kHz)
PRE_ROLL_CHUNKS = 3     # chunks to prepend before speech onset (~96 ms)


def record_audio(
    silence_duration: float | None = None,
    min_speech_frames: int | None = None,
    pre_roll_chunks: int | None = None,
    vad_threshold: float | None = None,
) -> np.ndarray:
    """
    Block until speech is detected, record until silence, return audio array.
    Returns a 1-D int16 numpy array at SAMPLE_RATE, or empty array if
    the captured audio was too short to be real speech.
    """
    print("[recorder] Listening for speech...")

    active_silence_duration = SILENCE_DURATION if silence_duration is None else silence_duration
    active_min_speech_frames = MIN_SPEECH_FRAMES if min_speech_frames is None else min_speech_frames
    active_pre_roll_chunks = PRE_ROLL_CHUNKS if pre_roll_chunks is None else pre_roll_chunks

    pre_roll: collections.deque = collections.deque(maxlen=active_pre_roll_chunks)
    recorded: list[np.ndarray] = []
    silence_frames  = 0
    speech_frames   = 0
    speech_started  = False
    silence_limit   = int(active_silence_duration * SAMPLE_RATE / CHUNK_FRAMES)

    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype="int16",
                        blocksize=CHUNK_FRAMES) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_FRAMES)
            chunk = chunk[:, 0]          # flatten to 1-D

            if detect_voice_activity(chunk, threshold=vad_threshold):
                if not speech_started:
                    rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                    print(f"[recorder] Speech detected (RMS={rms:.0f}) — recording...")
                    speech_started = True
                    # Prepend pre-roll so we don't clip the first syllable
                    recorded.extend(pre_roll)
                silence_frames = 0
                speech_frames += 1
                recorded.append(chunk)
            elif speech_started:
                recorded.append(chunk)   # keep trailing silence for natural end
                silence_frames += 1
                if silence_frames >= silence_limit:
                    print("[recorder] Silence detected — stopping.")
                    break
            else:
                # Still in pre-speech phase — maintain rolling pre-roll buffer
                pre_roll.append(chunk)

    if not recorded:
        return np.array([], dtype=np.int16)

    # Discard recordings that are too short to be real speech (noise burst guard)
    if speech_frames < active_min_speech_frames:
        print(f"[recorder] Discarded — only {speech_frames} voiced frames "
              f"(need {active_min_speech_frames}). Likely noise.")
        return np.array([], dtype=np.int16)

    return np.concatenate(recorded)