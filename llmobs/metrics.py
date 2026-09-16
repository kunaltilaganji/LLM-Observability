"""Prometheus metrics.

Histogram bucket choice is the only interesting decision here. Prometheus
computes quantiles by interpolating within a bucket, so a P95 is only as
precise as the bucket it lands in -- default buckets top out at 10s, which
would put every cold-start and every 8B-model request in the +Inf bucket and
report a P95 of "at least 10 seconds". The buckets below are chosen from
Project 2's measured latency distribution on this task, so the interesting part
of the curve is where the resolution is.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

# Extraction on a 3-4B model runs ~1-10s warm; cold start and 8B builds push
# past 30s. Resolution is concentrated in 1-15s where the SLO lives.
#
# The 120s and 300s buckets were added after the first incident run, where the
# top finite bucket was 60s: `histogram_quantile` cannot report a value above
# the highest finite bound, so the dashboard read a flat 60.0s P95 while
# callers were actually waiting 170s. A P95 sitting exactly on a bucket
# boundary is a saturated metric, not a measurement -- the buckets bound what
# is observable, and an incident that exceeds them is invisible at its peak.
_LATENCY_BUCKETS = (
    0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0,
    60.0, 120.0, 300.0,
)
# TTFT is dominated by prompt eval; it is much tighter than total latency and
# needs its own scale or every value lands in one bucket.
_TTFT_BUCKETS = (
    0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
)
# Queue wait is zero under light load and explodes under concurrency -- that
# bimodality is the whole point of measuring it separately.
_QUEUE_BUCKETS = (
    0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
)

_LBL = ("model", "schema", "mode")
_LBL_OUTCOME = _LBL + ("outcome",)


# --------------------------------------------------------------------------
# SRE: latency and throughput
# --------------------------------------------------------------------------
request_duration = Histogram(
    "llmobs_request_duration_seconds",
    "End-to-end latency from request arrival to response, including queueing and retries.",
    _LBL_OUTCOME,
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

# Deliberately separate from request_duration rather than a replacement for it.
# request_duration - generation_duration is the queueing, and keeping both
# means a latency regression can be attributed without opening a trace: if
# generation moved, it is the model; if only the total moved, it is the queue.
generation_duration = Histogram(
    "llmobs_generation_duration_seconds",
    "Time spent generating once a model slot was held; excludes queueing.",
    _LBL,
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

ttft = Histogram(
    "llmobs_ttft_seconds",
    "Time to first content token on the first attempt.",
    _LBL,
    buckets=_TTFT_BUCKETS,
    registry=REGISTRY,
)

queue_wait = Histogram(
    "llmobs_queue_wait_seconds",
    "Time spent waiting for a concurrency slot before the model was called.",
    _LBL,
    buckets=_QUEUE_BUCKETS,
    registry=REGISTRY,
)

# Ollama reports eval_duration, which excludes queueing and HTTP. The gap
# between this and request_duration is the part a server-side metric cannot
# see -- and is exactly what the post-mortem incident is about.
server_eval_duration = Histogram(
    "llmobs_server_eval_duration_seconds",
    "Decode time as reported by Ollama; excludes queueing, HTTP, and weight loading.",
    _LBL,
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

tokens_total = Counter(
    "llmobs_tokens_total",
    "Tokens processed, split by direction.",
    _LBL + ("direction",),
    registry=REGISTRY,
)

inflight = Gauge(
    "llmobs_inflight_requests",
    "Requests currently holding or waiting for a model slot.",
    registry=REGISTRY,
)


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------
requests_total = Counter(
    "llmobs_requests_total",
    "Requests handled, by terminal outcome.",
    _LBL_OUTCOME,
    registry=REGISTRY,
)

attempts_total = Counter(
    "llmobs_attempts_total",
    "Model invocations, including corrective retries.",
    _LBL,
    registry=REGISTRY,
)

retries_total = Counter(
    "llmobs_retries_total",
    "Corrective retries issued after a rejected first attempt.",
    _LBL,
    registry=REGISTRY,
)

retry_recovered_total = Counter(
    "llmobs_retry_recovered_total",
    "Retries that produced a valid record where the first attempt failed.",
    _LBL,
    registry=REGISTRY,
)

failures_total = Counter(
    "llmobs_failures_total",
    "Terminal failures by kind (json_parse_error, truncated, transport_error, ...).",
    _LBL + ("kind",),
    registry=REGISTRY,
)

fallbacks_total = Counter(
    "llmobs_fallbacks_total",
    "Requests answered with the degraded fallback instead of an extracted record.",
    _LBL + ("reason",),
    registry=REGISTRY,
)

# Project 2 found that `format: <schema>` does not guarantee bare JSON -- models
# still emit fenced output. A client trusting the flag silently discards valid
# records, so the repair rate is tracked as a first-class reliability metric.
extraction_repairs_total = Counter(
    "llmobs_extraction_repairs_total",
    "Responses that required client-side JSON extraction despite a format constraint.",
    _LBL,
    registry=REGISTRY,
)


# --------------------------------------------------------------------------
# Quality (graded traffic only)
# --------------------------------------------------------------------------
# Only requests carrying a known task_id can be scored, so these are populated
# by the canary/replay path rather than by every production request. That is
# the honest arrangement: online quality metrics need labels, and labels do not
# exist for organic traffic.
graded_fields_total = Counter(
    "llmobs_graded_fields_total",
    "Gold-labelled fields scored, by result (correct/incorrect).",
    _LBL + ("result",),
    registry=REGISTRY,
)

graded_errors_total = Counter(
    "llmobs_graded_errors_total",
    "Field-level errors by direction: commission (invented) vs omission (dropped).",
    _LBL + ("direction",),
    registry=REGISTRY,
)

field_accuracy = Gauge(
    "llmobs_field_accuracy_ratio",
    "Mean field accuracy over the most recent graded window.",
    _LBL,
    registry=REGISTRY,
)


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------
gpu_seconds_total = Counter(
    "llmobs_gpu_seconds_total",
    "GPU-seconds consumed serving requests.",
    _LBL,
    registry=REGISTRY,
)

cost_usd_total = Counter(
    "llmobs_cost_usd_total",
    "Serving cost in USD, by accounting basis: 'gpu' (actual) or a hosted-API counterfactual.",
    _LBL + ("basis",),
    registry=REGISTRY,
)

vram_peak_mib = Gauge(
    "llmobs_vram_peak_mib",
    "Peak device VRAM observed during the most recent request.",
    ("model",),
    registry=REGISTRY,
)


def observe_outcome_labels(model: str, schema: str, mode: str) -> dict[str, str]:
    return {"model": model, "schema": schema, "mode": mode}
