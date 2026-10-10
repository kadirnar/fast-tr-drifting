"""Silence between sentences by their final punctuation, measured per voice (``pause="punct"``).

``Synthesizer`` joins the sentences of a text with these pauses (release v3.2), or with a fixed number of seconds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .text import PUNCTUATION

# Pauses at internal marks in the training data (energy-based silence at the mark: mean, std in seconds) and the
# silence the model already leaves at the edges of a generated sentence (``edge``: leading + trailing, T 0.3, α 2,
# vocos-ft). Measured on 1000 utterances per group.
# Marks seen fewer than 20 times are left out (they fall back to ".").
PAUSES: dict[int | str, dict] = {
    722: {"edge": 0.161, ".": (0.125, 0.093), "?": (0.103, 0.064), ",": (0.052, 0.053)},  # studio
    389: {"edge": 0.176, ".": (0.848, 0.348), ",": (0.473, 0.208)},  # male
    323: {"edge": 0.070, ".": (0.223, 0.328), ",": (0.126, 0.174)},  # female
    "base": {"edge": 0.123, ".": (0.234, 0.329), "?": (0.238, 0.326), ",": (0.159, 0.190)},  # corpus without 722
}


@dataclass
class PausePolicy:
    """Silence between sentences by the first one's final mark: a ``pause`` of ``Synthesizer``.

    The gap between two generated sentences is the trailing silence of the first, the inserted silence and the
    leading silence of the second. The policy aims at a gap like the training data's pause at that mark (``gaps``:
    mark -> ``(mean, std)`` seconds; ``jitter`` 0 gives the mean, ``j > 0`` draws ``mean + j * std * N(0, 1)``) and
    inserts ``max(0, gap - edge)``, ``edge`` being the sentences' own edge silence. A sentence cut without a mark (a
    long one split at a space) uses the ``,`` entry; a mark missing from ``gaps`` uses ``.``."""

    gaps: dict[str, tuple[float, float]] = field(default_factory=lambda: {".": (0.15, 0.0)})
    edge: float = 0.0
    jitter: float = 0.0

    @classmethod
    def for_voice(cls, speaker: int | str | None = None, jitter: float = 0.0, edge: float | None = None) -> PausePolicy:
        """The measured policy of a speaker ID (else of the multi-speaker corpus, ``base``). ``edge``: the edge
        silence of the pipeline that generates the sentences, if it differs from the table's (measured with v3.1's
        regressors; a prosody predictor's sampled durations leave other edges, ``ProsodyPredictor.pause_edges``)."""
        table = dict(PAUSES.get(int(speaker) if str(speaker).isdigit() else speaker, PAUSES["base"]))
        table_edge = table.pop("edge")
        return cls(table, table_edge if edge is None else edge, jitter)

    def __call__(self, sentence: str, rng: random.Random | None = None) -> float:
        mark = sentence.rstrip()[-1:]
        mean, std = self.gaps.get(mark if mark in PUNCTUATION else ",", self.gaps["."])
        gap = mean + self.jitter * std * rng.gauss(0.0, 1.0) if self.jitter and rng is not None else mean
        return float(max(0.0, gap - self.edge))
