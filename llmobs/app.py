"""FastAPI extraction service.

Project 2's specification called for a FastAPI wrapper around the local model
(step 1.2) and the study shipped without one -- a batch harness needs no
serving path. This is that service, built so there is something real to
observe: a monitoring layer bolted onto a benchmark script would have no
concurrency, no queueing, and no incidents to diagnose.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import metrics as M
from . import tracing as T
from .cost import HOSTED_RATES
from .pipeline import ExtractionService
from .settings import SETTINGS
from .slm_bridge import SCHEMAS, TASKS_BY_ID

_service: ExtractionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    T.init_tracing()
    _service = ExtractionService()
    try:
        yield
    finally:
        T.flush()
        if _service is not None:
            _service.close()


app = FastAPI(
    title="Offline SLM Extraction Service",
    version="1.0.0",
    description=(
        "Structured extraction served from a local model, instrumented with "
        "OpenTelemetry traces and Prometheus metrics."
    ),
    lifespan=lifespan,
)


class ExtractRequest(BaseModel):
    text: str = Field(description="The unstructured source text.")
    schema_name: Literal["action_item", "support_ticket", "invoice"] = Field(
        description="Which Pydantic schema to extract into."
    )
    model: str | None = Field(default=None, description="Ollama tag; defaults to config.")
    mode: Literal["prompt", "json", "schema"] | None = Field(
        default=None,
        description="Enforcement mode. `schema` guarantees shape, not truth.",
    )
    # Graded traffic: a caller may name a task from Project 2's gold suite, or
    # supply labels inline. Organic traffic carries neither and is simply not
    # scored -- online quality metrics require labels, and pretending otherwise
    # is how dashboards end up reporting accuracy nobody measured.
    task_id: str | None = Field(default=None, description="Gold task id, for canary traffic.")
    gold: dict[str, Any] | None = Field(default=None, description="Inline gold labels.")
    temperature: float | None = None
    seed: int | None = None
    num_predict: int | None = None


class ExtractResponse(BaseModel):
    request_id: str
    trace_id: str
    ok: bool
    record: dict[str, Any] | None
    fallback: bool
    failure: str
    model: str
    mode: str
    attempts: int
    recovered_by_retry: bool
    latency_ms: float
    generate_ms: float
    queue_wait_ms: float
    ttft_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    gpu_seconds: float
    cost_usd: float
    hosted_cost_usd: float
    grade: dict[str, Any] | None
    warnings: list[str]


def _svc() -> ExtractionService:
    if _service is None:  # pragma: no cover
        raise HTTPException(503, "service not ready")
    return _service


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    if req.schema_name not in SCHEMAS:
        raise HTTPException(400, f"unknown schema {req.schema_name!r}")
    result = await _svc().extract(
        text=req.text,
        schema=req.schema_name,
        model=req.model,
        mode=req.mode,
        task_id=req.task_id,
        gold=req.gold,
        temperature=req.temperature,
        seed=req.seed,
        num_predict=req.num_predict,
    )
    # A failed extraction is a 200 carrying an explicit abstention, not a 5xx.
    # The request was served correctly; the model declined to produce a record
    # that validates. Returning 500 would fold a known, expected outcome into
    # the same bucket as a crashed process on every dashboard downstream.
    return ExtractResponse(
        request_id=result.request_id,
        trace_id=result.trace_id,
        ok=result.ok,
        record=result.record,
        fallback=result.fallback,
        failure=result.failure,
        model=result.model,
        mode=result.mode,
        attempts=result.attempts,
        recovered_by_retry=result.recovered_by_retry,
        latency_ms=result.latency_ms,
        generate_ms=result.generate_ms,
        queue_wait_ms=result.queue_wait_ms,
        ttft_ms=result.ttft_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        gpu_seconds=result.gpu_seconds,
        cost_usd=result.cost_usd,
        hosted_cost_usd=result.hosted_cost_usd,
        grade=result.grade,
        warnings=result.warnings,
    )


@app.get("/v1/schemas")
async def schemas() -> dict[str, Any]:
    return {
        name: {
            "json_schema": cls.model_json_schema(),
            "gold_tasks": [t for t, task in TASKS_BY_ID.items() if task.schema == name],
        }
        for name, cls in SCHEMAS.items()
    }


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Liveness plus the config that determines behaviour.

    The settings are echoed because the incident case study works by changing
    them on a running service; a health check that does not report what the
    service is currently configured to do is not much of a health check.
    """
    svc = _svc()
    try:
        local = sorted(svc.client.list_local())
        loaded = [m.get("name") for m in svc.client.loaded()]
        reachable = True
    except Exception as e:  # pragma: no cover
        local, loaded, reachable = [], [], False
        return {"ok": False, "ollama_reachable": reachable, "error": str(e)}
    return {
        "ok": True,
        "ollama_reachable": reachable,
        "models_available": local,
        "models_loaded": loaded,
        "config": {
            "default_model": SETTINGS.default_model,
            "default_mode": SETTINGS.default_mode,
            "max_concurrency": SETTINGS.max_concurrency,
            "max_retries": SETTINGS.max_retries,
            "num_predict": SETTINGS.default_num_predict,
            "fallback_deadline_s": SETTINGS.fallback_deadline_s,
            "gpu_hourly_usd": SETTINGS.gpu_hourly_usd,
            "hosted_comparison": {
                k: {
                    "model": r.model_id,
                    "input_per_mtok_usd": r.input_per_mtok_usd,
                    "output_per_mtok_usd": r.output_per_mtok_usd,
                }
                for k, r in HOSTED_RATES.items()
            },
        },
        "slo": {
            "p95_latency_s": SETTINGS.slo_p95_latency_s,
            "schema_validity": SETTINGS.slo_schema_validity,
            "field_accuracy": SETTINGS.slo_field_accuracy,
        },
    }


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(generate_latest(M.REGISTRY), media_type=CONTENT_TYPE_LATEST)
