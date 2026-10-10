"""Public text sets for benchmarks and checks: Freya-TR-Eval (495 everyday Turkish sentences, CC-BY-4.0)."""

from __future__ import annotations

import json
from pathlib import Path

FREYA = "freyavoice/freya-tr-eval"


def load_texts(spec: str = FREYA) -> list[dict]:
    """``[{"id", "text", ...}]`` from a HF dataset id (its ``freya_tr_eval.jsonl``), a ``.jsonl`` (field ``text``) or a
    ``.txt`` file (one sentence per line)."""
    path = Path(spec)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(spec, "freya_tr_eval.jsonl", repo_type="dataset"))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{"id": f"{i:04d}", "text": t.strip()} for i, t in enumerate(path.read_text().splitlines()) if t.strip()]


def request_mix(spec: str, n: int, every: int = 4, sentences: int = 3) -> list[str]:
    """``n`` request texts: the set's items in order, every ``every``-th request a paragraph of ``sentences`` items
    (the mix of docs/LATENCY.md: for n = 256, 192 sentences and 64 paragraphs)."""
    items = [r["text"] for r in load_texts(spec)]
    out, k = [], 0
    for i in range(n):
        m = sentences if every and i % every == every - 1 else 1
        out.append(" ".join(items[(k + j) % len(items)] for j in range(m)))
        k += m
    return out
