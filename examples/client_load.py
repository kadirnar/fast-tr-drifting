"""Load test for examples/server.py: N WebSocket clients send their requests at the same instant.

Each client opens its connection first; when all are connected they send their request together and record, on the
client side, the time to the first audio message (TTFA: network, JSON, PCM encoding and the event loops included),
the time to the last one, and whether playback from the first piece on would have stalled. The requests are the mix
of benchmarks/bench_concurrency.py (Freya-TR-Eval sentences in order, every fourth a paragraph of three; client
``i`` uses seed ``i``).

    python examples/client_load.py --clients 1 64 256 --url ws://127.0.0.1:8000/tts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np

from fast_tr_drifting.audio import SAMPLE_RATE
from fast_tr_drifting.evalsets import FREYA, request_mix


async def client(url: str, request: dict, connected: list, ready: asyncio.Event, start: asyncio.Event) -> dict:
    import websockets

    async with websockets.connect(url, max_size=1 << 24, open_timeout=120, ping_interval=None,
                                  compression=None) as ws:
        connected.append(1)
        if len(connected) == connected[0] + 1:  # the first entry holds the number of clients
            ready.set()
        await start.wait()
        t0 = time.perf_counter()
        await ws.send(json.dumps(request))
        events, done = [], None
        while done is None:
            msg = await ws.recv()
            t = time.perf_counter() - t0
            if isinstance(msg, bytes):
                events.append((t, len(msg) // 2))
            else:
                done = json.loads(msg)
                if "error" in done:
                    raise RuntimeError(done["error"])
        return {"events": events, "done": t, "server_ttfa_ms": done.get("server_ttfa_ms")}


def stalls(events: list[tuple[float, int]]) -> tuple[float, float]:
    """Seconds a client playing from its first piece on would wait for audio, and the least audio still buffered."""
    play, total, least = events[0][0] + events[0][1] / SAMPLE_RATE, 0.0, float("inf")
    for t, n in events[1:]:
        least = min(least, play - t)
        if t > play:
            total, play = total + t - play, t
        play += n / SAMPLE_RATE
    return total, least


async def run(url: str, texts: list[str], voice: str) -> list[dict]:
    connected, ready, start = [len(texts)], asyncio.Event(), asyncio.Event()
    tasks = [asyncio.create_task(client(url, {"text": t, "voice": voice, "seed": i}, connected, ready, start))
             for i, t in enumerate(texts)]
    waiter = asyncio.create_task(ready.wait())  # every client connected: they all send at once
    await asyncio.wait([waiter, *tasks], return_when=asyncio.FIRST_COMPLETED)
    for t in tasks:
        if t.done() and t.exception() is not None:  # a client could not connect
            waiter.cancel()
            raise t.exception()
    await waiter
    await asyncio.sleep(0.2)
    start.set()
    return await asyncio.gather(*tasks)


def summarize(results: list[dict]) -> dict:
    ttfa = np.array([r["events"][0][0] for r in results]) * 1000
    server = np.array([r["server_ttfa_ms"] for r in results if r["server_ttfa_ms"] is not None])
    play = np.array([stalls(r["events"]) for r in results]) * 1000
    audio = sum(n for r in results for _, n in r["events"]) / SAMPLE_RATE
    done = max(r["done"] for r in results)
    pct = {"p50": np.median(ttfa), "p90": np.percentile(ttfa, 90), "max": ttfa.max()}
    return {"clients": len(results), "ttfa_ms": {k: round(float(v), 1) for k, v in pct.items()},
            "server_ttfa_ms_p50": round(float(np.median(server)), 1) if len(server) else None,
            "all_done_ms": round(1000 * done, 1), "audio_s": round(audio, 1), "audio_per_s": round(audio / done, 1),
            "stalled": int((play[:, 0] > 0).sum()), "buffer_ms_min": round(float(play[:, 1].min()), 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="ws://127.0.0.1:8000/tts")
    p.add_argument("--clients", type=int, nargs="+", default=[1, 64, 256])
    p.add_argument("--repeats", type=int, default=3, help="timed runs per N (after one warm-up run)")
    p.add_argument("--voice", default="studio")
    p.add_argument("--texts", default=FREYA)
    p.add_argument("--out", default=None, help="write the results as JSON")
    args = p.parse_args()
    texts = request_mix(args.texts, max(args.clients))
    out = {}
    print(f"{'clients':>7s} {'TTFA p50':>9s} {'p90':>8s} {'max':>8s} {'server p50':>10s} {'all done':>9s} "
          f"{'audio/s':>8s} {'stalled':>7s} {'buffer':>7s}")
    for n in args.clients:
        asyncio.run(run(args.url, texts[:n], args.voice))  # warm-up
        runs = [summarize(asyncio.run(run(args.url, texts[:n], args.voice))) for _ in range(args.repeats)]
        out[n] = runs
        for r in runs:
            t = r["ttfa_ms"]
            print(f"{n:7d} {t['p50']:7.1f}ms {t['p90']:6.1f}ms {t['max']:6.1f}ms {r['server_ttfa_ms_p50']:8.1f}ms "
                  f"{r['all_done_ms']:7.0f}ms {r['audio_per_s']:8.1f} {r['stalled']:7d} {r['buffer_ms_min']:7.0f}",
                  flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
