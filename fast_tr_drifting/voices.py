"""The voices of the released model: the best-rated of its training speakers, by integer speaker ID.

``male`` and ``female`` were chosen on 30 held-out ``dev`` sentences per candidate (UTMOSv2 and Whisper large-v3 CER),
never on a test set: the best male and the best female voice of the 24 best-covered speakers. ``studio`` (the
default) was added later by fine-tuning (speaker 722, checkpoint v3.1). Other training speaker IDs still load, but
these are the supported, documented voices.
"""

from __future__ import annotations

VOICES: dict[str, dict] = {
    "male": {"id": 389, "pitch_hz": 104, "train_minutes": 59},
    "female": {"id": 323, "pitch_hz": 174, "train_minutes": 19},
    # added by fine-tuning on one studio voice (configs/acoustic_v3_add_voice.yaml); the clearest voice, the default
    "studio": {"id": 722, "pitch_hz": 103, "train_minutes": 2058},
}
DEFAULT_VOICE = "studio"

# the vocoder each voice sounds best with (Freya-495 UTMOSv2 with v3.2: male 2.896 -> 2.944, female 2.722 -> 2.749
# with the speaker-balanced Vocos v2; the studio voice keeps Vocos v2): Synthesizer.from_pretrained(vocoder="per-voice")
VOICE_VOCODERS: dict[int, str] = {722: "vocos-v2", 389: "vocos-v2-balanced", 323: "vocos-v2-balanced"}


def voice_id(voice: str | int) -> int:
    """Speaker ID of a voice name (``studio`` / ``male`` / ``female``) or of a raw training speaker ID."""
    if isinstance(voice, int) or str(voice).isdigit():
        return int(voice)
    if voice not in VOICES:
        raise KeyError(f"unknown voice {voice!r}; choose one of {', '.join(VOICES)}")
    return VOICES[voice]["id"]
