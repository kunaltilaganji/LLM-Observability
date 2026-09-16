#!/usr/bin/env python
"""Regression gate over Project 2's gold suite.

Grades the running service against the same 52 hand-labelled tasks Project 2's
report is built on, and fails the build when quality, reliability, or latency
drifts below the recorded baseline. Reusing that suite rather than writing a
fresh one is the point: the labels are already published and already argued
with, so the gate measures drift in the *service*, not in a private rubric.

Three modes:

  --record    run live, write eval/baseline.json and eval/fixtures/
  live        run live, compare against the recorded baseline
  --replay    re-score stored fixtures with no service and no GPU

`--replay` is what runs on a GitHub-hosted runner, and it is worth being exact
about what it proves: it re-runs the *scoring and gating* logic against frozen
model outputs, so a bug in the scorer, the thresholds, or the gate arithmetic
fails the build. It does not re-run the model, so it cannot catch a model or
config regression. Only the live job on a self-hosted runner does that. A CI
badge that implies otherwise would be decorative.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmobs.settings import SETTINGS  # noqa: E402
from llmobs.slm_bridge import ALL_TASKS, score_record  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
BASELINE = EVAL_DIR / "baseline.json"
FIXTURES = EVAL_DIR / "fixtures"


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100 * len(s))) - 1))
    return s[k]


def run_live(base_url: str, model: str | None, mode: str | None,
             save_fixtures: bool) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v1/extract"
    rows: list[dict[str, Any]] = []
    if save_fixtures:
        FIXTURES.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=300.0) as client:
        for i, task in enumerate(ALL_TASKS, 1):
            payload: dict[str, Any] = {
                "text": task.text,
                "schema_name": task.schema,
                "task_id": task.id,
            }
            if model:
                payload["model"] = model
            if mode:
                payload["mode"] = mode
            t0 = time.perf_counter()
            r = client.post(url, json=payload)
            r.raise_for_status()
            body = r.json()
            row = {
                "task_id": task.id,
                "schema": task.schema,
                "tags": list(task.tags),
                "ok": body["ok"],
                "failure": body["failure"],
                "record": body["record"],
                "latency_s": time.perf_counter() - t0,
                "attempts": body["attempts"],
                "model": body["model"],
                "mode": body["mode"],
            }
            rows.append(row)
            if save_fixtures:
                (FIXTURES / f"{task.id}.json").write_text(json.dumps(row, indent=2))
            print(f"  [{i:>2}/{len(ALL_TASKS)}] {task.id:<8} "
                  f"{'ok ' if body['ok'] else 'FAIL'} {row['latency_s']:>6.2f}s")
    return rows


def load_fixtures() -> list[dict[str, Any]]:
    if not FIXTURES.exists():
        raise SystemExit("no fixtures recorded; run with --record against a live service")
    return [json.loads(p.read_text()) for p in sorted(FIXTURES.glob("*.json"))]


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate, and break accuracy out by failure-mode tag.

    The per-tag split matters more than the headline: 'accuracy fell 4 points'
    does not tell an on-call engineer anything, while 'null-handling collapsed
    while arithmetic held' points straight at the change.
    """
    # Accumulated as lists rather than a single value per task: keying a plain
    # dict by task_id means a second row for the same task silently replaces
    # the first, so a suite run twice (or a fixture directory with duplicates)
    # would report only the last result and quietly discard the rest.
    by_task_scores: dict[str, list[float]] = {}
    tag_scores: dict[str, list[float]] = {}
    commissions = omissions = graded_fields = 0
    valid = 0
    latencies: list[float] = []
    failures: dict[str, int] = {}

    gold_by_id = {t.id: t.gold for t in ALL_TASKS}

    for row in rows:
        latencies.append(row["latency_s"])
        if not row["ok"]:
            failures[row["failure"]] = failures.get(row["failure"], 0) + 1
            # A request that produced no valid record scores zero rather than
            # being dropped. Excluding failures would let a model that refuses
            # half the suite post a higher accuracy than one that answers it.
            by_task_scores.setdefault(row["task_id"], []).append(0.0)
            for tag in row.get("tags", []):
                tag_scores.setdefault(tag, []).append(0.0)
            continue
        valid += 1
        gold = gold_by_id[row["task_id"]]
        mean, per_field = score_record(row["record"], gold)
        by_task_scores.setdefault(row["task_id"], []).append(mean)
        for tag in row.get("tags", []):
            tag_scores.setdefault(tag, []).append(mean)
        for k, s in per_field.items():
            graded_fields += 1
            if s == 1.0:
                continue
            g, p = gold.get(k), row["record"].get(k)
            if g is None and p is not None:
                commissions += 1
            elif g is not None and p is None:
                omissions += 1

    n = len(rows) or 1
    by_task = {k: statistics.mean(v) for k, v in by_task_scores.items()}
    return {
        "n_tasks": len(rows),
        "field_accuracy": round(statistics.mean(by_task.values()), 4) if by_task else 0.0,
        "schema_validity": round(valid / n, 4),
        "commission_rate": round(commissions / graded_fields, 4) if graded_fields else 0.0,
        "omission_rate": round(omissions / graded_fields, 4) if graded_fields else 0.0,
        "latency_p50_s": round(_pct(latencies, 50), 3),
        "latency_p95_s": round(_pct(latencies, 95), 3),
        "failures": failures,
        "by_tag": {
            t: round(statistics.mean(v), 4) for t, v in sorted(tag_scores.items())
        },
        "by_task": {k: round(v, 4) for k, v in sorted(by_task.items())},
        "model": rows[0].get("model") if rows else None,
        "mode": rows[0].get("mode") if rows else None,
    }


def compare(current: dict[str, Any], baseline: dict[str, Any],
            tolerance: float, check_latency: bool) -> list[str]:
    """Return the list of gate violations. Empty means pass."""
    v: list[str] = []

    acc_floor = baseline["field_accuracy"] - tolerance
    if current["field_accuracy"] < acc_floor:
        v.append(
            f"field accuracy {current['field_accuracy']:.1%} is below the "
            f"baseline {baseline['field_accuracy']:.1%} by more than the "
            f"{tolerance:.0%} tolerance"
        )
    if current["field_accuracy"] < SETTINGS.slo_field_accuracy:
        v.append(
            f"field accuracy {current['field_accuracy']:.1%} is below the "
            f"absolute SLO floor {SETTINGS.slo_field_accuracy:.0%}"
        )
    if current["schema_validity"] < SETTINGS.slo_schema_validity:
        v.append(
            f"schema validity {current['schema_validity']:.1%} is below the "
            f"SLO floor {SETTINGS.slo_schema_validity:.0%}"
        )
    # Latency is only gated on a live run. Fixture latencies are whatever the
    # machine that recorded them managed, so asserting on them in replay would
    # be gating on a number CI never measured.
    if check_latency and current["latency_p95_s"] > SETTINGS.slo_p95_latency_s:
        v.append(
            f"P95 latency {current['latency_p95_s']:.2f}s exceeds the SLO "
            f"{SETTINGS.slo_p95_latency_s:.1f}s"
        )

    # Per-tag regressions catch the case the aggregate hides: one failure mode
    # collapsing while the mean stays inside tolerance.
    for tag, base_score in baseline.get("by_tag", {}).items():
        cur = current.get("by_tag", {}).get(tag)
        if cur is None:
            continue
        if cur < base_score - max(tolerance * 2, 0.10):
            v.append(
                f"failure mode '{tag}' regressed {base_score:.0%} -> {cur:.0%}"
            )
    return v


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8100")
    p.add_argument("--model", default=None)
    p.add_argument("--mode", default=None, choices=["prompt", "json", "schema"])
    p.add_argument("--record", action="store_true",
                   help="write baseline.json and fixtures/ instead of gating")
    p.add_argument("--replay", action="store_true",
                   help="score stored fixtures; no service or GPU required")
    p.add_argument("--tolerance", type=float, default=SETTINGS.gate_accuracy_tolerance,
                   help="allowed accuracy drop below baseline, in points (0.03 = 3pp)")
    p.add_argument("--report", default=None, help="write the scored report here")
    args = p.parse_args()

    if args.replay:
        print("Replaying stored fixtures (scoring logic only -- the model is not re-run)")
        rows = load_fixtures()
    else:
        print(f"Grading {len(ALL_TASKS)} gold tasks against {args.base_url}")
        rows = run_live(args.base_url, args.model, args.mode, save_fixtures=args.record)

    current = score(rows)

    print("\n" + "=" * 62)
    print(f"  tasks              {current['n_tasks']}")
    print(f"  field accuracy     {current['field_accuracy']:.1%}")
    print(f"  schema validity    {current['schema_validity']:.1%}")
    print(f"  omission rate      {current['omission_rate']:.1%}")
    print(f"  commission rate    {current['commission_rate']:.1%}")
    print(f"  latency P50 / P95  {current['latency_p50_s']:.2f}s / {current['latency_p95_s']:.2f}s")
    if current["failures"]:
        print(f"  failures           {current['failures']}")
    print("=" * 62)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(current, indent=2))
        print(f"report -> {args.report}")

    if args.record:
        BASELINE.write_text(json.dumps(current, indent=2))
        print(f"\nbaseline recorded -> {BASELINE}")
        print("Commit this file. Subsequent runs gate against it.")
        return 0

    if not BASELINE.exists():
        print("\nno baseline recorded; run with --record first", file=sys.stderr)
        return 2

    baseline = json.loads(BASELINE.read_text())
    violations = compare(current, baseline, args.tolerance,
                         check_latency=not args.replay)

    if violations:
        print("\nGATE FAILED")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(f"\nGATE PASSED (baseline accuracy {baseline['field_accuracy']:.1%}, "
          f"tolerance {args.tolerance:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
