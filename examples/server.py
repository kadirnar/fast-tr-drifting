"""A streaming TTS server with dynamic micro-batching (FastAPI; WebSocket and HTTP chunked streaming).

One GPU worker thread takes requests from a queue. When it is free and a request arrives, it keeps collecting
requests until none has arrived for ``--window-ms`` (at most ``--max-wait-ms`` after the first, at most
``--max-batch``), then streams the whole group: a lone request through the single-request fast path (CUDA graphs), a
group through :func:`fast_tr_drifting.batched.stream_batched` (one voice per batch; requests are grouped by voice).
Every round's pieces go back to their requests as 16-bit PCM as soon as they are on the host (one hand-over to the
event loop per round).
All paths use the streaming DiT with counter-based noise (``Chunking(noise="philox")``), so a request sounds the same
alone or in any batch. Requests that arrive while a group is streaming wait for the next group (no continuous
batching).

    pip install "fast-tr-drifting[serve]"
    python examples/server.py --port 8000             # compiles the batched passes first (about 1.5 min)

Protocol:

* ``ws://host:port/tts``: send ``{"text": "...", "voice": "studio", "seed": 0}`` (JSON); the server answers with
  binary messages (PCM16, little-endian, mono, 24 kHz) and a final ``{"done": true, ...}``; the connection can be
  reused for the next request;
* ``POST /tts`` with the same JSON: the PCM16 stream as a chunked HTTP response (``audio/L16; rate=24000``);
* ``GET /health``: the server's settings and counters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass, field

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from fast_tr_drifting.audio import SAMPLE_RATE, to_pcm16
from fast_tr_drifting.batched import Serving, stream_batched
from fast_tr_drifting.chunked import Chunking
from fast_tr_drifting.synthesize import Synthesizer
from fast_tr_drifting.voices import VOICES

WARMUP = ["Merhaba, nasılsınız?", "Toplantı yarın saat 14:30'da başlayacak. Lütfen raporlarınızı getirin.",
          "İstanbul Boğazı'nın iki yakası, 1973 yılında açılan köprüyle birbirine bağlandı.", "Evet."]


@dataclass
class Job:
    text: str
    voice: str
    seed: int
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue  # bytes (a piece), then None (done) or an Exception
    arrived: float = field(default_factory=time.perf_counter)
    first: float | None = None


def _dispatch(items: list[tuple[Job, object]]) -> None:
    for job, item in items:
        job.out.put_nowait(item)


class Batcher(threading.Thread):
    """The GPU worker: dynamic micro-batches from a queue (see the module docstring)."""

    def __init__(self, synth: Synthesizer, window_ms: float, max_batch: int, serving: Serving, cfg: float,
                 temperature: float, warmup: bool = True, max_wait_ms: float = 30.0):
        super().__init__(daemon=True)
        self.do_warmup = warmup
        self.synth, self.window, self.max_batch, self.serving = synth, window_ms / 1000, max_batch, serving
        self.max_wait = max_wait_ms / 1000
        self.kw = dict(cfg_scale=cfg, temperature=temperature, chunked=Chunking(noise="philox"))
        self.jobs: queue.Queue[Job] = queue.Queue()
        self.stats = {"requests": 0, "batches": 0, "largest_batch": 0}
        self.ready = threading.Event()

    def submit(self, job: Job) -> None:
        self.jobs.put(job)

    def _send(self, items: list[tuple[Job, object]]) -> None:
        """Hand pieces (bytes), ``None`` (done) or an exception to their requests' event loops: one call per loop."""
        now, by_loop = time.perf_counter(), {}
        for job, item in items:
            if job.first is None and isinstance(item, bytes):
                job.first = now
            by_loop.setdefault(job.loop, []).append((job, item))
        for loop, batch in by_loop.items():
            loop.call_soon_threadsafe(_dispatch, batch)

    def _collect(self) -> list[Job]:
        """The first request, then every one that follows within ``window`` of the one before (a burst), for at most
        ``max_wait`` and ``max_batch`` requests."""
        batch = [self.jobs.get()]
        end = time.perf_counter() + self.max_wait
        while len(batch) < self.max_batch:
            try:
                batch.append(self.jobs.get(timeout=max(0.0, min(self.window, end - time.perf_counter()))))
            except queue.Empty:
                break
        return batch

    @torch.no_grad()
    def _stream(self, group: list[Job]) -> None:
        voice = group[0].voice
        if len(group) == 1:  # the single-request fast path: CUDA graphs
            job = group[0]
            for piece in self.synth.stream(job.text, speaker=voice, seed=job.seed, **self.kw):
                self._send([(job, to_pcm16(piece))])
        else:
            rounds = stream_batched(self.synth, [j.text for j in group], speaker=voice, seeds=[j.seed for j in group],
                                    serving=self.serving, **self.kw)
            for pieces in rounds:
                self._send([(group[i], to_pcm16(piece)) for i, piece in pieces])

    def run(self) -> None:
        if self.do_warmup:  # in this thread: the one that will replay the graphs and run the compiled passes
            t = time.perf_counter()
            self.warmup()
            print(f"warmed up in {time.perf_counter() - t:.0f} s", flush=True)
        self.ready.set()
        while True:
            batch = self._collect()
            self.stats["requests"] += len(batch)
            by_voice: dict[str, list[Job]] = {}
            for job in batch:
                by_voice.setdefault(job.voice, []).append(job)
            for group in by_voice.values():
                self.stats["batches"] += 1
                self.stats["largest_batch"] = max(self.stats["largest_batch"], len(group))
                try:
                    self._stream(group)
                    self._send([(job, None) for job in group])
                except Exception as e:  # noqa: BLE001 - report to the clients, keep serving
                    self._send([(job, e) for job in group])

    @torch.no_grad()
    def warmup(self) -> None:
        """Run everything a request can meet once: the fast path of every voice, and the batched passes (compiled
        with ``Serving.fast()``) for batches of several rows and of one row (a batch's last rounds can have one row
        left), with every voice's vocoder."""
        for voice in VOICES:
            list(self.synth.stream(WARMUP[0], speaker=voice, **self.kw))
            for n in (len(WARMUP), 2):
                for _ in stream_batched(self.synth, WARMUP[:n], speaker=voice, serving=self.serving, **self.kw):
                    pass
        torch.cuda.synchronize()


def build_app(batcher: Batcher, max_chars: int) -> FastAPI:
    # FastAPI reads the handlers' annotations (WebSocket, dict): the names must be importable from this module
    app = FastAPI(title="fast-tr-drifting")

    def make_job(msg: dict) -> Job:
        text = " ".join(str(msg.get("text", "")).split())
        voice = str(msg.get("voice", "studio"))
        if not text or len(text) > max_chars:
            raise ValueError(f"text must have 1 to {max_chars} characters")
        if voice not in VOICES:
            raise ValueError(f"voice must be one of {', '.join(VOICES)}")
        return Job(text, voice, int(msg.get("seed", 0)), asyncio.get_running_loop(), asyncio.Queue())

    async def pieces(job: Job):
        batcher.submit(job)
        while True:
            item = await job.out.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    @app.get("/health")
    async def health():
        return {"status": "ok", "window_ms": batcher.window * 1000, "max_wait_ms": batcher.max_wait * 1000,
                "max_batch": batcher.max_batch,
                "serving": str(batcher.serving), **batcher.stats}

    @app.post("/tts")
    async def tts_http(msg: dict):
        try:
            job = make_job(msg)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return StreamingResponse(pieces(job), media_type=f"audio/L16; rate={SAMPLE_RATE}; channels=1")

    @app.websocket("/tts")
    async def tts_ws(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                try:
                    job = make_job(json.loads(await ws.receive_text()))
                except ValueError as e:
                    await ws.send_text(json.dumps({"error": str(e)}))
                    continue
                n = 0
                try:
                    async for pcm in pieces(job):
                        await ws.send_bytes(pcm)
                        n += len(pcm) // 2
                except Exception as e:  # noqa: BLE001
                    await ws.send_text(json.dumps({"error": f"{type(e).__name__}: {e}"}))
                    continue
                await ws.send_text(json.dumps({
                    "done": True, "samples": n, "seconds": n / SAMPLE_RATE,
                    "server_ttfa_ms": None if job.first is None else round(1000 * (job.first - job.arrived), 2)}))
        except WebSocketDisconnect:
            return

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--window-ms", type=float, default=4.0,
                   help="a free worker collects requests until none has arrived for this long")
    p.add_argument("--max-wait-ms", type=float, default=30.0, help="... and at most this long after the first")
    p.add_argument("--max-batch", type=int, default=256)
    p.add_argument("--serving", choices=["fast", "fp32"], default="fast",
                   help="fast: Serving.fast() (fp16 + torch.compile); fp32: the batched passes as shipped")
    p.add_argument("--vocoder", default="per-voice", help="per-voice | vocos-v2 | vocos-v2-balanced | a checkpoint")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-chars", type=int, default=1000)
    args = p.parse_args()

    import uvicorn

    synth = Synthesizer.from_pretrained("v3.2", "cuda", fast=True, vocoder=args.vocoder)
    serving = Serving.fast() if args.serving == "fast" else Serving()
    batcher = Batcher(synth, args.window_ms, args.max_batch, serving, args.cfg, args.temperature,
                      max_wait_ms=args.max_wait_ms)
    batcher.start()
    batcher.ready.wait()
    # no per-message deflate: compressing raw PCM costs CPU time and saves little
    uvicorn.run(build_app(batcher, args.max_chars), host=args.host, port=args.port, log_level="warning",
                ws_max_size=1 << 20, ws_per_message_deflate=False)


if __name__ == "__main__":
    main()
