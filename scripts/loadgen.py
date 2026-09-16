#!/usr/bin/env python
"""Traffic generator.

Dashboards built on synthetic uniform traffic look nothing like dashboards
built on real traffic, and the difference is the whole point of percentiles.
Two arrival models are supported:

  closed   N workers, each sending the next request as soon as the last
           returns. Self-limiting -- queue depth cannot grow, so P95 stays
           flat no matter how slow the service gets. This is what most naive
           load scripts do, and it is why they never reproduce a queueing
           incident.

  open     Requests are issued on a Poisson schedule at a fixed rate,
           regardless of whether earlier ones have returned. If the service
           cannot keep up, the backlog grows without bound -- which is exactly
           the failure mode a P95 alert exists to catch.

Default is `open`, because the incident case study needs it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmobs.slm_bridge import ALL_TASKS  # noqa: E402


async def _one(client: httpx.AsyncClient, url: str, task, mode: str | None,
               model: str | None, graded: bool) -> dict:
    payload = {
        "text": task.text,
        "schema_name": task.schema,
        # Sending the task id is what makes the request gradeable; the service
        # looks the gold labels up rather than trusting anything the client
        # sends. Ungraded traffic exercises the same path without polluting the
        # quality metrics with unlabelled data.
        "task_id": task.id if graded else None,
    }
    if mode:
        payload["mode"] = mode
    if model:
        payload["model"] = model
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=180.0)
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}", "latency_s": elapsed}
        body = r.json()
        return {
            "ok": body["ok"],
            "task_id": task.id,
            "failure": body["failure"],
            # Two latencies, deliberately both kept. `latency_s` is what this
            # client waited; `server_latency_ms` is what the service believes it
            # took. Anything the service cannot see -- connection-pool waits,
            # socket backlog, the network -- lives in the gap between them, and
            # a load test that records only one of the two cannot find it.
            "latency_s": elapsed,
            "server_latency_ms": body["latency_ms"],
            "generate_ms": body["generate_ms"],
            "queue_wait_ms": body["queue_wait_ms"],
            "attempts": body["attempts"],
            "accuracy": (body.get("grade") or {}).get("field_accuracy"),
            "cost_usd": body["cost_usd"],
            "hosted_cost_usd": body["hosted_cost_usd"],
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "latency_s": time.perf_counter() - t0}


async def run(args: argparse.Namespace) -> int:
    url = f"{args.base_url.rstrip('/')}/v1/extract"
    tasks = [t for t in ALL_TASKS if not args.schema or t.schema == args.schema]
    if not tasks:
        print(f"no tasks for schema {args.schema!r}", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)

    results: list[dict] = []
    # The client's own connection pool is a queue too. Left at httpx's default
    # this becomes the binding constraint before the service does, and the run
    # measures the load generator rather than the service under test -- which
    # is precisely what happened on the first attempt at the incident below.
    limits = httpx.Limits(
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_connections,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        started = time.perf_counter()
        pending: set[asyncio.Task] = set()

        if args.mode == "closed":
            async def worker(wid: int) -> None:
                i = 0
                while time.perf_counter() - started < args.duration:
                    task = tasks[(wid * 7 + i) % len(tasks)]
                    results.append(await _one(client, url, task, args.format_mode,
                                              args.model, not args.ungraded))
                    i += 1
            await asyncio.gather(*(worker(w) for w in range(args.concurrency)))
        else:
            # Poisson arrivals: exponential inter-arrival gaps. A fixed sleep
            # would produce a metronome, which smooths away exactly the burst
            # behaviour that creates queueing.
            i = 0
            while time.perf_counter() - started < args.duration:
                task = tasks[i % len(tasks)]
                t = asyncio.create_task(
                    _one(client, url, task, args.format_mode, args.model,
                         not args.ungraded)
                )
                pending.add(t)
                t.add_done_callback(lambda f: (pending.discard(f),
                                               results.append(f.result())))
                i += 1
                await asyncio.sleep(rng.expovariate(args.rate))
            if pending:
                print(f"waiting on {len(pending)} in-flight requests...")
                await asyncio.gather(*pending, return_exceptions=True)

    return _report(results, args)


def _report(results: list[dict], args: argparse.Namespace) -> int:
    if not results:
        print("no results", file=sys.stderr)
        return 1
    lat = sorted(r["latency_s"] for r in results)
    ok = [r for r in results if r.get("ok")]
    graded = [r["accuracy"] for r in results if r.get("accuracy") is not None]

    def pct(p: float) -> float:
        # Nearest-rank rather than interpolation: with a few hundred samples,
        # interpolating invents a latency no request actually had.
        k = max(0, min(len(lat) - 1, int(round(p / 100 * len(lat))) - 1))
        return lat[k]

    print(f"\n{'-' * 62}")
    print(f"  arrival model     {args.mode}"
          + (f" @ {args.rate}/s" if args.mode == "open" else f" x{args.concurrency}"))
    print(f"  requests          {len(results)}  ({len(ok)} ok, "
          f"{len(results) - len(ok)} failed/fallback)")
    print(f"  client latency    P50 {pct(50):>8.2f}s   P90 {pct(90):>8.2f}s   "
          f"P95 {pct(95):>8.2f}s   max {lat[-1]:>8.2f}s")

    # The same percentiles from the service's own point of view. When these
    # diverge the difference is time the service never saw -- and no amount of
    # server-side instrumentation will surface it.
    srv = sorted(r["server_latency_ms"] / 1000 for r in results
                 if r.get("server_latency_ms") is not None)
    if srv:
        def spct(p: float) -> float:
            k = max(0, min(len(srv) - 1, int(round(p / 100 * len(srv))) - 1))
            return srv[k]
        print(f"  server latency    P50 {spct(50):>8.2f}s   P90 {spct(90):>8.2f}s   "
              f"P95 {spct(95):>8.2f}s   max {srv[-1]:>8.2f}s")
        gap = pct(95) - spct(95)
        if gap > 0.5:
            print(f"  ** {gap:.1f}s of the P95 is invisible to the service "
                  f"({gap / pct(95):.0%} of what the caller waited) **")

    gen = [r["generate_ms"] for r in results if r.get("generate_ms") is not None]
    qw = [r.get("queue_wait_ms", 0) for r in results if "queue_wait_ms" in r]
    if gen and qw:
        print(f"  server-side split  generation mean {statistics.mean(gen):>8.0f}ms   "
              f"slot wait mean {statistics.mean(qw):>8.0f}ms (max {max(qw):.0f}ms)")
    if graded:
        print(f"  field accuracy    {statistics.mean(graded):.1%}  (n={len(graded)})")
    gpu = sum(r.get("cost_usd", 0) for r in results)
    hosted = sum(r.get("hosted_cost_usd", 0) for r in results)
    if gpu or hosted:
        print(f"  cost              ${gpu:.5f} gpu  vs  ${hosted:.5f} hosted")
    kinds: dict[str, int] = {}
    for r in results:
        if not r.get("ok"):
            kinds[r.get("failure") or r.get("error", "?")] = (
                kinds.get(r.get("failure") or r.get("error", "?"), 0) + 1
            )
    if kinds:
        print(f"  failures          {kinds}")
    print(f"{'-' * 62}\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"raw results -> {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8100")
    p.add_argument("--mode", choices=["open", "closed"], default="open",
                   help="arrival model (default: open, which can build a backlog)")
    p.add_argument("--rate", type=float, default=1.0, help="requests/sec (open mode)")
    p.add_argument("--concurrency", type=int, default=4, help="workers (closed mode)")
    p.add_argument("--duration", type=float, default=60.0, help="seconds")
    p.add_argument("--schema", default=None,
                   choices=["action_item", "support_ticket", "invoice"])
    p.add_argument("--model", default=None)
    p.add_argument("--format-mode", dest="format_mode", default=None,
                   choices=["prompt", "json", "schema"])
    p.add_argument("--ungraded", action="store_true",
                   help="omit task ids, so quality metrics are not populated")
    p.add_argument("--max-connections", type=int, default=512,
                   help="client connection-pool size; keep well above the "
                        "in-flight count or the generator throttles itself")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None, help="write raw per-request JSON here")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
