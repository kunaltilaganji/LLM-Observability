"""OpenTelemetry wiring.

Instrumentation is written against the OTel SDK plus OpenInference semantic
conventions rather than a vendor SDK. That is the whole reason this layer is
portable: Phoenix is the backend that happens to run here, but the same spans
export to any OTLP collector, and swapping backends is an endpoint change, not
a rewrite.

Two exporters run side by side:

  * OTLP -> Phoenix, for interactive trace inspection.
  * JSONL -> results/traces/, because a reviewer cloning this repo cannot start
    my Phoenix instance. The evidence behind the post-mortem has to survive
    outside the tool that produced it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from .settings import SETTINGS

try:
    from openinference.semconv.trace import (
        OpenInferenceSpanKindValues,
        SpanAttributes,
    )

    _OI = True
except ImportError:  # pragma: no cover - openinference is a hard dep in practice
    _OI = False


# --------------------------------------------------------------------------
# Attribute names
# --------------------------------------------------------------------------
# OpenInference names are used where they exist so Phoenix renders spans as
# first-class LLM calls (token counts, prompt/response panes) instead of opaque
# blobs. Everything this project measures that OpenInference has no opinion
# about -- TTFT, VRAM, retry provenance, queue wait -- is namespaced under
# `llmobs.` rather than squatted into the standard namespace.

if _OI:
    SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
    LLM_MODEL = SpanAttributes.LLM_MODEL_NAME
    LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
    LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
    LLM_INVOCATION_PARAMS = SpanAttributes.LLM_INVOCATION_PARAMETERS
    TOKENS_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
    TOKENS_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
    TOKENS_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
    INPUT_VALUE = SpanAttributes.INPUT_VALUE
    OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
    INPUT_MIME = SpanAttributes.INPUT_MIME_TYPE
    OUTPUT_MIME = SpanAttributes.OUTPUT_MIME_TYPE
    KIND_CHAIN = OpenInferenceSpanKindValues.CHAIN.value
    KIND_LLM = OpenInferenceSpanKindValues.LLM.value
    KIND_TOOL = OpenInferenceSpanKindValues.TOOL.value
else:  # pragma: no cover
    SPAN_KIND = "openinference.span.kind"
    LLM_MODEL = "llm.model_name"
    LLM_PROVIDER = "llm.provider"
    LLM_SYSTEM = "llm.system"
    LLM_INVOCATION_PARAMS = "llm.invocation_parameters"
    TOKENS_PROMPT = "llm.token_count.prompt"
    TOKENS_COMPLETION = "llm.token_count.completion"
    TOKENS_TOTAL = "llm.token_count.total"
    INPUT_VALUE = "input.value"
    OUTPUT_VALUE = "output.value"
    INPUT_MIME = "input.mime_type"
    OUTPUT_MIME = "output.mime_type"
    KIND_CHAIN, KIND_LLM, KIND_TOOL = "CHAIN", "LLM", "TOOL"

# Project-specific attributes.
A_TTFT_MS = "llmobs.ttft_ms"
A_TTFT_ANY_MS = "llmobs.ttft_any_ms"
A_WALL_TPS = "llmobs.wall_tps"
A_SERVER_TPS = "llmobs.server_tps"
A_LOAD_MS = "llmobs.load_duration_ms"
A_EVAL_MS = "llmobs.eval_duration_ms"
A_UNACCOUNTED_MS = "llmobs.unaccounted_ms"
A_QUEUE_WAIT_MS = "llmobs.queue_wait_ms"
A_VRAM_PEAK = "llmobs.vram_peak_mib"
A_VRAM_DELTA = "llmobs.vram_delta_mib"
A_COLD_START = "llmobs.cold_start"
A_ATTEMPT = "llmobs.attempt"
A_FAILURE_KIND = "llmobs.failure_kind"
A_FORMAT_MODE = "llmobs.format_mode"
A_NEEDED_EXTRACTION = "llmobs.needed_extraction"
A_RECOVERED = "llmobs.recovered_by_retry"
A_THINKING_CHARS = "llmobs.thinking_chars"
A_SCHEMA = "llmobs.schema"
A_TASK_ID = "llmobs.task_id"
A_FIELD_ACCURACY = "llmobs.field_accuracy"
A_OMISSIONS = "llmobs.omissions"
A_COMMISSIONS = "llmobs.commissions"
A_COST_USD = "llmobs.cost_usd"
A_COST_GPU_S = "llmobs.gpu_seconds"
A_FALLBACK = "llmobs.fallback"
A_FINISH_REASON = "llmobs.finish_reason"


class JsonlSpanExporter(SpanExporter):
    """Append finished spans to a JSONL file.

    Deliberately writes flat rows rather than the OTLP protobuf shape: the point
    is that `results/traces/spans.jsonl` can be read with pandas by anyone, with
    no collector running and no protobuf dependency.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows = []
        for s in spans:
            ctx = s.get_span_context()
            rows.append(
                {
                    "name": s.name,
                    "trace_id": f"{ctx.trace_id:032x}",
                    "span_id": f"{ctx.span_id:016x}",
                    "parent_span_id": (
                        f"{s.parent.span_id:016x}" if s.parent else None
                    ),
                    "start_time_ns": s.start_time,
                    "end_time_ns": s.end_time,
                    "duration_ms": (
                        (s.end_time - s.start_time) / 1e6
                        if s.end_time and s.start_time
                        else None
                    ),
                    "status": s.status.status_code.name if s.status else None,
                    "status_description": s.status.description if s.status else None,
                    "attributes": dict(s.attributes or {}),
                    "events": [
                        {"name": e.name, "timestamp_ns": e.timestamp,
                         "attributes": dict(e.attributes or {})}
                        for e in s.events
                    ],
                }
            )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


_provider: TracerProvider | None = None
_init_lock = threading.Lock()


def init_tracing(
    *, service_name: str | None = None, jsonl_path: Path | None = None
) -> TracerProvider:
    """Idempotently install the global tracer provider."""
    global _provider
    with _init_lock:
        if _provider is not None:
            return _provider

        resource = Resource.create(
            {
                "service.name": service_name or SETTINGS.service_name,
                "service.version": "1.0.0",
                # Recorded on every span because the whole point of Project 2's
                # findings is that they are hardware-specific. A trace that does
                # not say which GPU produced it is not reproducible evidence.
                "host.name": _hostname(),
                "gpu.model": _gpu_name(),
            }
        )
        provider = TracerProvider(resource=resource)

        if SETTINGS.tracing_enabled:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                provider.add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(endpoint=SETTINGS.otlp_endpoint)
                    )
                )
            except Exception as e:  # pragma: no cover
                # A dead collector must never take the service down with it.
                # Observability is not allowed to become a new failure mode.
                print(f"[tracing] OTLP exporter disabled: {e}")

        path = jsonl_path or (SETTINGS.trace_export_dir / "spans.jsonl")
        provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))

        trace.set_tracer_provider(provider)
        _provider = provider
        return provider


def get_tracer(name: str = "llmobs") -> trace.Tracer:
    init_tracing()
    return trace.get_tracer(name)


def flush() -> None:
    if _provider is not None:
        _provider.force_flush(timeout_millis=5000)


def truncate(value: Any, limit: int | None = None) -> str:
    limit = limit or SETTINGS.payload_char_limit
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    return s if len(s) <= limit else s[:limit] + f"…[+{len(s) - limit} chars]"


def set_payload(span: trace.Span, key: str, value: Any, mime_key: str | None = None) -> None:
    """Attach prompt/response text, honouring the capture toggle."""
    if not SETTINGS.capture_payloads:
        return
    span.set_attribute(key, truncate(value))
    if mime_key:
        span.set_attribute(mime_key, "application/json" if not isinstance(value, str) else "text/plain")


def _hostname() -> str:
    import socket

    return socket.gethostname()


def _gpu_name() -> str:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(SETTINGS.vram_device_index)
        name = pynvml.nvmlDeviceGetName(h)
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return "unknown"
