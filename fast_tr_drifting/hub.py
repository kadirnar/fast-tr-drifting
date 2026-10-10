"""Published files: the release table of :data:`HUB_REPO` and the named vocoders and prosody predictors.

A release is an acoustic model, a vocoder, a prosody source and a pause rule
(:meth:`fast_tr_drifting.synthesize.Synthesizer.from_pretrained`). Files are downloaded from the Hub at a pinned
revision, or read from a local directory laid out like the repo (``$FAST_TR_DRIFTING_HUB_DIR``) when it holds them.
"""

from __future__ import annotations

import os
from pathlib import Path

HUB_REPO = "Vyvo/drifting-tts-tr"
# the model repo's commit with every file of release v3.2 (its weights are CC-BY-4.0); None: the latest
HUB_REVISION = os.environ.get("FAST_TR_DRIFTING_REVISION", "0906c46992410a2032c6f5777b887f8f1780a6fd")
HUB_DIR_ENV = "FAST_TR_DRIFTING_HUB_DIR"

# name -> file of HUB_REPO
PROSODY_MODELS: dict[str, str] = {
    # stochastic prosody predictor trained with drifting on v3.1's text encoder: prosody temperature 0.5, per-voice
    # duration factors and pause edges stored in the checkpoint
    "drift": "prosody_drift_v3.2.pt",
}
VOCODERS: dict[str, str] = {
    "vocos-v2": "vocos_v2.pt",  # the vocoder of release v3.2 (13.5 M)
    "vocos-v2-balanced": "vocos_v2_balanced.pt",  # vocos-v2 trained 20k steps further on speaker-balanced batches
}

# release -> Synthesizer arguments; ``model`` is a file of HUB_REPO. v3.2 keeps v3.1's acoustic weights and samples
# only the token pitch: sampled durations cost intelligibility on new text for the voices with little data, so the
# durations stay the regressor's. The pauses between sentences follow the final punctuation, measured per voice.
RELEASES: dict[str, dict] = {
    "v3.2": {"model": "drifting_tts_v3.2.pt", "vocoder": "vocos-v2", "prosody": "drift",
             "prosody_durations": "regressor", "pause": "punct"},
}
LATEST = "v3.2"


def hub_file(filename: str, what: str | None = None) -> str:
    """Local path of ``filename`` of :data:`HUB_REPO`: ``$FAST_TR_DRIFTING_HUB_DIR/<filename>`` if that exists, else
    the Hub download (cached by ``huggingface_hub``). ``what`` names the caller's object in the error raised when the
    file cannot be found."""
    local = os.environ.get(HUB_DIR_ENV)
    if local and (Path(local) / filename).is_file():
        return str(Path(local) / filename)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(HUB_REPO, filename, revision=HUB_REVISION or None)
    except EntryNotFoundError as e:
        raise FileNotFoundError(f"{what or filename}: {HUB_REPO}/{filename} not found; pass a checkpoint path "
                                f"instead, or set {HUB_DIR_ENV}") from e


def resolve_prosody(spec: str | Path) -> str:
    """A prosody predictor name of :data:`PROSODY_MODELS` (e.g. ``drift``) or a checkpoint path -> a local path."""
    if str(spec) in PROSODY_MODELS:
        return hub_file(PROSODY_MODELS[str(spec)], f"prosody {spec!r}")
    if not Path(spec).is_file():
        raise ValueError(f"unknown prosody predictor {str(spec)!r}: expected a checkpoint path or one of "
                         f"{', '.join(PROSODY_MODELS)}")
    return str(spec)


def resolve_vocoder(spec: str | Path) -> str:
    """A vocoder name of :data:`VOCODERS` or a checkpoint path -> a local path."""
    if str(spec) in VOCODERS:
        return hub_file(VOCODERS[str(spec)], f"vocoder {spec!r}")
    if not Path(spec).is_file():
        raise ValueError(f"unknown vocoder {str(spec)!r}: expected a checkpoint path or one of {', '.join(VOCODERS)}")
    return str(spec)
