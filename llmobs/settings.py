"""Runtime configuration for the extraction service and its observability layer.

Everything an operator would plausibly want to change during an incident lives
here and is overridable by environment variable, because the incident case
study in `docs/post_mortem_case_study.md` works by changing exactly these
values on a running system and watching the dashboards react.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Project 2 lives in a sibling repository. It is imported by path rather than
# installed, because it was written as a study harness and never packaged; see
# llmobs/slm_bridge.py for the shim and the reasoning.
SLM_REPO = Path(
    os.getenv("SLM_REPO", str(REPO_ROOT.parent / "offline-slm-assistant"))
).resolve()


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # ---- serving ----
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    default_model: str = os.getenv("LLMOBS_MODEL", "gemma3:4b")
    default_mode: str = os.getenv("LLMOBS_MODE", "schema")  # prompt | json | schema
    default_num_predict: int = _i("LLMOBS_NUM_PREDICT", 512)
    default_temperature: float = _f("LLMOBS_TEMPERATURE", 0.0)
    default_seed: int = _i("LLMOBS_SEED", 42)
    max_retries: int = _i("LLMOBS_MAX_RETRIES", 1)

    # Requests queued behind a busy GPU are the most common source of a P95
    # blowout on a single-GPU deployment. The semaphore makes that queueing
    # *visible* (we time the wait) instead of letting it hide inside Ollama.
    max_concurrency: int = _i("LLMOBS_MAX_CONCURRENCY", 4)
    request_timeout_s: float = _f("LLMOBS_TIMEOUT_S", 120.0)

    # Deadline after which we stop waiting and return the degraded fallback
    # rather than a 500. An abstention a user can act on beats a timeout.
    fallback_deadline_s: float = _f("LLMOBS_FALLBACK_DEADLINE_S", 25.0)

    # ---- observability ----
    service_name: str = os.getenv("OTEL_SERVICE_NAME", "llm-extraction-service")
    otlp_endpoint: str = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006/v1/traces"
    )
    tracing_enabled: bool = _b("LLMOBS_TRACING", True)

    # Prompt and completion text are recorded on spans by default because this
    # is a single-user offline deployment and the payloads are the whole point
    # of a trace. Anything handling real user data would flip this off.
    capture_payloads: bool = _b("LLMOBS_CAPTURE_PAYLOADS", True)
    payload_char_limit: int = _i("LLMOBS_PAYLOAD_CHARS", 4000)

    vram_device_index: int = _i("LLMOBS_GPU_INDEX", 0)

    # ---- cost ----
    # $/hour for the GPU this runs on. Local inference has no per-token bill,
    # so the honest unit cost is amortised hardware time. Default is the
    # on-demand rental rate for an RTX A6000 -- override for owned hardware.
    gpu_hourly_usd: float = _f("LLMOBS_GPU_HOURLY_USD", 0.79)

    # ---- quality gate thresholds (shared by alerts and CI) ----
    slo_p95_latency_s: float = _f("LLMOBS_SLO_P95_S", 8.0)
    slo_schema_validity: float = _f("LLMOBS_SLO_SCHEMA_VALIDITY", 0.98)
    slo_field_accuracy: float = _f("LLMOBS_SLO_FIELD_ACCURACY", 0.75)
    # How far below the recorded baseline a gated run may drift, in points.
    gate_accuracy_tolerance: float = _f("LLMOBS_GATE_TOLERANCE", 0.03)

    prometheus_port: int = _i("LLMOBS_PROM_PORT", 9090)
    service_port: int = _i("LLMOBS_PORT", 8100)

    # ---- storage ----
    trace_export_dir: Path = field(default_factory=lambda: REPO_ROOT / "results" / "traces")
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "results")


SETTINGS = Settings()
