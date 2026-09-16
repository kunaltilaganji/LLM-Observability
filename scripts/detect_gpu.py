#!/usr/bin/env python
"""Work out which GPU Ollama actually places a model on.

The VRAM metric samples one device by index. On a shared multi-GPU host the
naive default -- device 0 -- is very likely wrong: this service was built on a
box where devices 0 and 1 are saturated by another tenant and Ollama places its
models on 2 or 3. Sampling device 0 there would report 46 GB of somebody else's
memory as this service's footprint, which is worse than reporting nothing.

Ollama does not expose the device it chose, so this measures it: snapshot every
device, load the model, snapshot again, and take the device whose used memory
grew. Prints the index on stdout so `scripts/stack.sh` can export it.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

try:
    import pynvml
except ImportError:  # pragma: no cover
    print("0")
    sys.exit(0)


def snapshot(handles) -> list[float]:
    out = []
    for h in handles:
        try:
            out.append(pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2))
        except Exception:
            out.append(0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
    except Exception as e:
        log(f"NVML unavailable ({e}); defaulting to device 0")
        print("0")
        return 0

    with httpx.Client(timeout=300.0) as c:
        # Evict first so the delta is attributable. If the model is already
        # resident the load is a no-op and every device reads flat, which would
        # silently fall through to device 0.
        try:
            c.post(f"{args.host}/api/chat",
                   json={"model": args.model, "messages": [], "keep_alive": 0})
            time.sleep(3)
        except Exception:
            pass

        before = snapshot(handles)
        try:
            c.post(f"{args.host}/api/chat", json={
                "model": args.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "keep_alive": "30m",
                "options": {"num_predict": 1},
            })
        except Exception as e:
            log(f"could not load {args.model} ({e}); defaulting to device 0")
            print("0")
            return 0
        time.sleep(3)
        after = snapshot(handles)

    deltas = [a - b for a, b in zip(after, before)]
    idx = max(range(n), key=lambda i: deltas[i])
    if deltas[idx] < 200:
        log(f"no device grew by >200 MiB (deltas={[round(d) for d in deltas]}); "
            f"defaulting to device 0")
        print("0")
        return 0

    log(f"{args.model} loaded on GPU {idx} (+{deltas[idx]:.0f} MiB; "
        f"deltas={[round(d) for d in deltas]})")
    print(str(idx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
