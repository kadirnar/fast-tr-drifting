"""examples/server.py on a tiny random model (CPU): the WebSocket and HTTP streams, micro-batches and errors. Needs
the ``serve`` extra (FastAPI) and ``httpx``."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from fast_tr_drifting.audio import to_pcm16  # noqa: E402
from fast_tr_drifting.batched import Serving  # noqa: E402
from fast_tr_drifting.chunked import Chunking  # noqa: E402
from fast_tr_drifting.synthesize import Synthesizer  # noqa: E402

from .test_batched import MODEL, _tiny_vocos  # noqa: E402

TEXTS = ["Merhaba! Bugün hava çok güzel.", "Bu bir deneme.", "Yapay zekâ modelleri her geçen gün hızlanıyor."]


def _server_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "server.py"
    spec = importlib.util.spec_from_file_location("fast_tr_drifting_server_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from fast_tr_drifting.models.tts import DriftingTTS

    tmp = tmp_path_factory.mktemp("server")
    torch.manual_seed(0)
    model = DriftingTTS(MODEL, num_speakers=723).eval()  # the voices' speaker IDs: 722, 389, 323
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    torch.save({"ema": model.state_dict(), "config": {"model": MODEL}, "num_speakers": 723,
                "stats": {"mean": -5.0, "std": 2.0, "backend": "bigvgan"}}, tmp / "tts.pt")
    synth = Synthesizer(tmp / "tts.pt", "cpu", vocoder=_tiny_vocos(), pause="punct")
    server = _server_module()
    batcher = server.Batcher(synth, window_ms=50, max_batch=8, serving=Serving(), cfg=2.0, temperature=0.3,
                             warmup=False)
    batcher.start()
    batcher.ready.wait()
    return server, synth, batcher, TestClient(server.build_app(batcher, max_chars=200))


def _alone(synth, text: str, voice: str, seed: int) -> np.ndarray:
    pieces = synth.stream(text, speaker=voice, seed=seed, cfg_scale=2.0, temperature=0.3,
                          chunked=Chunking(noise="philox"))
    return np.frombuffer(b"".join(to_pcm16(p) for p in pieces), dtype="<i2")


def test_websocket_stream(setup):
    _, synth, _, client = setup
    with client.websocket_connect("/tts") as ws:
        for seed, voice in ((3, "studio"), (4, "female")):  # one connection, two requests
            ws.send_text(json.dumps({"text": TEXTS[0], "voice": voice, "seed": seed}))
            pcm = b""
            while True:
                msg = ws.receive()
                if msg.get("bytes") is not None:
                    pcm += msg["bytes"]
                    continue
                done = json.loads(msg["text"])
                break
            got = np.frombuffer(pcm, dtype="<i2")
            ref = _alone(synth, TEXTS[0], voice, seed)
            assert done["done"] and done["samples"] == len(got) == len(ref)
            assert np.abs(got.astype(int) - ref).max() <= 1  # the same audio as the request alone
        ws.send_text(json.dumps({"text": "", "voice": "studio"}))
        assert "error" in json.loads(ws.receive_text())


def test_http_stream_and_health(setup):
    _, synth, _, client = setup
    r = client.post("/tts", json={"text": TEXTS[1], "voice": "male", "seed": 1})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/L16")
    got = np.frombuffer(r.content, dtype="<i2")
    assert np.abs(got.astype(int) - _alone(synth, TEXTS[1], "male", 1)).max() <= 1
    assert client.post("/tts", json={"text": "merhaba", "voice": "narrator"}).status_code == 400
    assert client.get("/health").json()["status"] == "ok"


def test_requests_arriving_together_form_one_batch(setup):
    server, synth, batcher, _ = setup
    before = dict(batcher.stats)

    async def run():
        loop = asyncio.get_running_loop()
        jobs = [server.Job(t, "studio", i, loop, asyncio.Queue()) for i, t in enumerate(TEXTS)]
        for job in jobs:  # within the 50 ms window: one micro-batch
            batcher.submit(job)

        async def drain(job):
            out = b""
            while (item := await job.out.get()) is not None:
                out += item
            return np.frombuffer(out, dtype="<i2")

        return await asyncio.gather(*(drain(j) for j in jobs))

    got = asyncio.run(run())
    assert batcher.stats["batches"] == before["batches"] + 1 and batcher.stats["largest_batch"] >= 3
    for i, text in enumerate(TEXTS):
        ref = _alone(synth, text, "studio", i)
        assert len(got[i]) == len(ref) and np.abs(got[i].astype(int) - ref).max() <= 1
