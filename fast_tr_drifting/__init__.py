"""fast-tr-drifting: one-step Turkish text-to-speech with drifting models (arXiv 2602.04770), built for speed.

    from fast_tr_drifting import Synthesizer
    tts = Synthesizer.from_pretrained("v3.2", "cuda", fast=True)
    for piece in tts.stream("Merhaba!", chunked=True):
        ...
"""

__version__ = "0.1.0"

__all__ = ["Chunking", "Serving", "Synthesizer", "stream_batched", "__version__"]


def __getattr__(name: str):
    """Lazy exports: importing the package does not import torch."""
    if name == "Synthesizer":
        from .synthesize import Synthesizer

        return Synthesizer
    if name == "Chunking":
        from .chunked import Chunking

        return Chunking
    if name in ("Serving", "stream_batched"):
        from . import batched

        return getattr(batched, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
