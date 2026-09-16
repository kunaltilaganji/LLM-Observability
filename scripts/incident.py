#!/usr/bin/env python
"""Drive the load incident and snapshot what the stack saw.

Runs three phases against the live service and records, at each one, the four
different answers the system gives to "how long did that take":

  1. what the caller waited            (client-side, from the load generator)
  2. what the service measured         (llmobs_request_duration_seconds)
  3. how long generation actually took (llmobs_generation_duration_seconds)
  4. what Ollama reported              (llmobs_server_eval_duration_seconds)

The gap between (2) and (4) is the case study. Everything is written to
reports/incident_timeline.json so the post-mortem quotes measurements rather
than recollections.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PROM = "http://localhost:9090"

QUERIES = {
    "client_p95_s": None,  # filled from the load generator, not Prometheus
    "request_p95_s": 'histogram_quantile(0.95, sum by(le) (rate(llmobs_request_duration_seconds_bucket[{w}])))',
    "request_p50_s": 'histogram_quantile(0.50, sum by(le) (rate(llmobs_request_duration_seconds_bucket[{w}])))',
    "queue_wait_p95_s": 'histogram_quantile(0.95, sum by(le) (rate(llmobs_queue_wait_seconds_bucket[{w}])))',
    "generation_p95_s": 'histogram_quantile(0.95, sum by(le) (rate(llmobs_generation_duration_seconds_bucket[{w}])))',
    "ollama_decode_p95_s": 'histogram_quantile(0.95, sum by(le) (rate(llmobs_server_eval_duration_seconds_bucket[{w}])))',
    "ttft_p95_s": 'histogram_quantile(0.95, sum by(le) (rate(llmobs_ttft_seconds_bucket[{w}])))',
    "peak_inflight": 'max_over_time(llmobs_inflight_requests[{w}])',
    "completed_per_min": 'sum(rate(llmobs_requests_total[{w}])) * 60',
    "fallback_rate": 'sum(rate(llmobs_fallbacks_total[{w}])) / clamp_min(sum(rate(llmobs_requests_total[{w}])), 1e-9)',
    "schema_validity": 'sum(rate(llmobs_requests_total{{outcome="ok"}}[{w}])) / clamp_min(sum(rate(llmobs_requests_total[{w}])), 1e-9)',
    "graded_accuracy": 'sum(rate(llmobs_graded_fields_total{{result="correct"}}[{w}])) / clamp_min(sum(rate(llmobs_graded_fields_total[{w}])), 1e-9)',
}


def scalar(client: httpx.Client, expr: str) -> float | None:
    r = client.get(f"{PROM}/api/v1/query", params={"query": expr})
    res = r.json().get("data", {}).get("result", [])
    if not res:
        return None
    return round(float(res[0]["value"][1]), 4)


def snapshot(client: httpx.Client, window: str) -> dict:
    out = {}
    for name, tmpl in QUERIES.items():
        if tmpl is None:
            continue
        try:
            out[name] = scalar(client, tmpl.format(w=window))
        except Exception as e:
            out[name] = f"error: {e}"
    return out


def alerts(client: httpx.Client) -> list[dict]:
    try:
        data = client.get(f"{PROM}/api/v1/alerts").json()["data"]["alerts"]
    except Exception:
        return []
    seen, out = set(), []
    for a in data:
        name = a["labels"]["alertname"]
        if name in seen:
            continue
        seen.add(name)
        out.append({
            "alert": name,
            "state": a["state"],
            "severity": a["labels"].get("severity"),
            "summary": a["annotations"].get("summary", ""),
        })
    return out


def drain(client: httpx.Client, host: str, model: str, budget_s: float = 900) -> float:
    """Wait until a trivial request is fast again.

    Necessary between phases: cancelling a client request does not cancel the
    generation behind it, so the GPU keeps working through the backlog long
    after the load generator has exited. Starting the next phase before that
    drains measures the previous phase.
    """
    deadline = time.time() + budget_s
    while time.time() < deadline:
        t0 = time.perf_counter()
        try:
            client.post(f"{host}/api/chat", json={
                "model": model, "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "options": {"num_predict": 2}, "keep_alive": "30m",
            }, timeout=120.0)
        except Exception:
            time.sleep(10)
            continue
        dt = time.perf_counter() - t0
        if dt < 1.0:
            return dt
        time.sleep(10)
    return -1.0


def run_phase(name: str, rate: float, duration: float, out_json: Path,
              python: str, extra: list[str]) -> dict:
    print(f"\n{'#' * 64}\n### {name}: {rate} req/s for {duration:.0f}s\n{'#' * 64}")
    cmd = [python, str(ROOT / "scripts" / "loadgen.py"),
           "--mode", "open", "--rate", str(rate), "--duration", str(duration),
           "--max-connections", "512", "--out", str(out_json)] + extra
    subprocess.run(cmd, check=False)
    rows = json.loads(out_json.read_text())
    lat = sorted(r["latency_s"] for r in rows)
    srv = sorted(r["server_latency_ms"] / 1000 for r in rows
                 if r.get("server_latency_ms") is not None)

    def p(s: list[float], q: float) -> float | None:
        if not s:
            return None
        return round(s[max(0, min(len(s) - 1, int(round(q / 100 * len(s))) - 1))], 3)

    return {
        "n": len(rows),
        "ok": sum(1 for r in rows if r.get("ok")),
        "client_p50_s": p(lat, 50), "client_p95_s": p(lat, 95),
        "client_max_s": round(lat[-1], 3) if lat else None,
        "server_p50_s": p(srv, 50), "server_p95_s": p(srv, 95),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--steady-rate", type=float, default=0.5)
    ap.add_argument("--spike-rate", type=float, default=4.0)
    ap.add_argument("--steady-duration", type=float, default=75)
    ap.add_argument("--spike-duration", type=float, default=90)
    args = ap.parse_args()

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    timeline: dict = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phases": []}

    with httpx.Client(timeout=60.0) as c:
        for label, rate, dur, window in [
            ("A_steady_before", args.steady_rate, args.steady_duration, "2m"),
            ("B_spike", args.spike_rate, args.spike_duration, "3m"),
            ("C_steady_after", args.steady_rate, args.steady_duration, "2m"),
        ]:
            print(f"\ndraining Ollama before {label}...")
            d = drain(c, args.ollama, args.model)
            print(f"  drained (probe {d:.2f}s)" if d > 0 else "  WARNING: did not drain")

            client_side = run_phase(label, rate, dur,
                                    reports / f"incident_{label}.json",
                                    args.python, [])
            # Snapshot immediately: Prometheus rate windows decay, and a
            # snapshot taken after the next phase starts describes neither.
            server_side = snapshot(c, window)
            fired = alerts(c)
            timeline["phases"].append({
                "phase": label, "rate_per_s": rate, "duration_s": dur,
                "prometheus_window": window,
                "client": client_side, "prometheus": server_side,
                "alerts_firing": fired,
            })
            print(f"\n  -- {label} recorded: "
                  f"client P95 {client_side['client_p95_s']}s | "
                  f"service P95 {server_side.get('request_p95_s')}s | "
                  f"ollama decode P95 {server_side.get('ollama_decode_p95_s')}s | "
                  f"{len(fired)} alert(s) firing")

    out = reports / "incident_timeline.json"
    out.write_text(json.dumps(timeline, indent=2))
    print(f"\ntimeline -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
