"""The instrumented extraction path.

The design constraint worth stating: Project 2's `StructuredGenerator` owns the
retry loop, and it stays that way. Re-implementing that loop here to get finer
spans would fork the logic the regression gate grades against, so instrumentation
is attached *underneath* it instead -- a traced subclass of the Ollama client
opens a span around every model call, and Project 2's loop nests inside it
without knowing anything about OpenTelemetry.

The consequence is honest span boundaries: `llm.generate` spans have real
timings because they wrap the real call. Parsing and validation happen inside
Project 2's loop where no span can be opened without forking it, so they are
recorded as span *events* on the request span, carrying the same failure kinds
the harness reports. Events describing real outcomes beat spans with invented
timestamps.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.trace import SpanKind, Status, StatusCode
from starlette.concurrency import run_in_threadpool

from . import metrics as M
from . import tracing as T
from .cost import DEFAULT_HOSTED, gpu_cost_usd, hosted_cost_usd
from .settings import SETTINGS
from .slm_bridge import (
    SCHEMAS,
    TASKS_BY_ID,
    OllamaClient,
    StructuredGenerator,
    score_record,
)

tracer = T.get_tracer("llmobs.pipeline")

# Attempt index for the current request, so the traced client can label its
# spans without the generator having to pass anything down.
_attempt_no: ContextVar[int] = ContextVar("llmobs_attempt_no", default=0)


class TracedOllamaClient(OllamaClient):
    """Project 2's client with a span around every model call.

    Subclassed rather than wrapped so the generator's own call sites are
    instrumented without modification -- including the corrective retry, which
    is the call an uninstrumented setup is most likely to miss.
    """

    def chat(self, model: str, messages: list[dict[str, str]], **kw: Any):  # type: ignore[override]
        attempt = _attempt_no.get()
        with tracer.start_as_current_span("llm.generate", kind=SpanKind.CLIENT) as span:
            span.set_attribute(T.SPAN_KIND, T.KIND_LLM)
            span.set_attribute(T.LLM_MODEL, model)
            span.set_attribute(T.LLM_PROVIDER, "ollama")
            span.set_attribute(T.LLM_SYSTEM, "ollama")
            span.set_attribute(T.A_ATTEMPT, attempt)
            span.set_attribute(
                T.LLM_INVOCATION_PARAMS,
                json.dumps(
                    {
                        "temperature": kw.get("temperature"),
                        "seed": kw.get("seed"),
                        "num_predict": kw.get("num_predict"),
                        "format": "schema" if isinstance(kw.get("fmt"), dict)
                        else (kw.get("fmt") or "none"),
                        "think": kw.get("think"),
                    },
                    default=str,
                ),
            )
            T.set_payload(span, T.INPUT_VALUE, messages, T.INPUT_MIME)

            gen = super().chat(model, messages, **kw)

            _attempt_no.set(attempt + 1)

            span.set_attribute(T.A_TTFT_MS, gen.ttft_ms or -1.0)
            span.set_attribute(T.A_TTFT_ANY_MS, gen.ttft_any_ms or -1.0)
            span.set_attribute(T.A_WALL_TPS, gen.wall_tps or -1.0)
            span.set_attribute(T.A_SERVER_TPS, gen.server_tps or -1.0)
            span.set_attribute(T.A_EVAL_MS, gen.eval_duration_ms or -1.0)
            span.set_attribute(T.A_LOAD_MS, gen.load_duration_ms or -1.0)
            span.set_attribute(T.A_FINISH_REASON, gen.finish_reason or "unknown")
            span.set_attribute(T.A_THINKING_CHARS, gen.thinking_chars)
            span.set_attribute(T.A_FORMAT_MODE, gen.format_mode)
            if gen.prompt_eval_count is not None:
                span.set_attribute(T.TOKENS_PROMPT, gen.prompt_eval_count)
            if gen.eval_count is not None:
                span.set_attribute(T.TOKENS_COMPLETION, gen.eval_count)
            if gen.prompt_eval_count is not None and gen.eval_count is not None:
                span.set_attribute(
                    T.TOKENS_TOTAL, gen.prompt_eval_count + gen.eval_count
                )
            if gen.peak_vram_mib is not None:
                span.set_attribute(T.A_VRAM_PEAK, gen.peak_vram_mib)
                span.set_attribute(T.A_VRAM_DELTA, gen.vram_delta_mib or 0.0)

            # The gap between what the user waited and what the server admits
            # to. Ollama's eval_duration excludes queueing, HTTP, and weight
            # loading; a dashboard built only on server-reported timings cannot
            # see a latency regression that lives in that gap.
            if gen.total_latency_ms is not None:
                accounted = (gen.eval_duration_ms or 0.0) + (
                    gen.prompt_eval_duration_ms or 0.0
                )
                span.set_attribute(
                    T.A_UNACCOUNTED_MS, round(gen.total_latency_ms - accounted, 2)
                )

            T.set_payload(span, T.OUTPUT_VALUE, gen.text, T.OUTPUT_MIME)
            if not gen.ok:
                span.set_status(Status(StatusCode.ERROR, gen.error or "transport error"))
            return gen


@dataclass
class ExtractionResult:
    request_id: str
    trace_id: str
    ok: bool
    record: dict[str, Any] | None
    schema: str
    model: str
    mode: str
    fallback: bool = False
    failure: str = "none"
    attempts: int = 0
    recovered_by_retry: bool = False
    latency_ms: float = 0.0
    generate_ms: float = 0.0
    queue_wait_ms: float = 0.0
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    gpu_seconds: float = 0.0
    cost_usd: float = 0.0
    hosted_cost_usd: float = 0.0
    grade: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


FALLBACK_RECORD = {
    "status": "extraction_unavailable",
    "detail": (
        "The model did not return a record that satisfies the schema. "
        "No partial or guessed values are returned."
    ),
}


class ExtractionService:
    """Serves structured extraction with full request-level instrumentation."""

    def __init__(self) -> None:
        self.client = TracedOllamaClient(
            host=SETTINGS.ollama_host, timeout_s=SETTINGS.request_timeout_s
        )
        self.generator = StructuredGenerator(
            self.client, max_retries=SETTINGS.max_retries
        )
        # Ollama serialises work per model anyway; the point of holding the
        # semaphore here is that the wait becomes *measurable* instead of
        # disappearing into the server's own queue where no client metric can
        # see it.
        self._slots = asyncio.Semaphore(SETTINGS.max_concurrency)

    def close(self) -> None:
        self.client.close()

    async def extract(
        self,
        *,
        text: str,
        schema: str,
        model: str | None = None,
        mode: str | None = None,
        task_id: str | None = None,
        gold: dict[str, Any] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
    ) -> ExtractionResult:
        model = model or SETTINGS.default_model
        mode = mode or SETTINGS.default_mode
        request_id = uuid.uuid4().hex[:16]
        labels = {"model": model, "schema": schema, "mode": mode}

        model_cls = SCHEMAS[schema]
        # A caller may pass gold labels directly, or name a task from Project
        # 2's suite. The second is how the canary and the CI gate grade traffic
        # without shipping the labels around.
        if gold is None and task_id and task_id in TASKS_BY_ID:
            gold = TASKS_BY_ID[task_id].gold

        with tracer.start_as_current_span("extract", kind=SpanKind.SERVER) as span:
            span.set_attribute(T.SPAN_KIND, T.KIND_CHAIN)
            span.set_attribute("llmobs.request_id", request_id)
            span.set_attribute(T.A_SCHEMA, schema)
            span.set_attribute(T.LLM_MODEL, model)
            span.set_attribute(T.A_FORMAT_MODE, mode)
            if task_id:
                span.set_attribute(T.A_TASK_ID, task_id)
            T.set_payload(span, T.INPUT_VALUE, text, T.INPUT_MIME)

            ctx = span.get_span_context()
            trace_id = f"{ctx.trace_id:032x}"

            t_enqueue = time.perf_counter()
            queue_wait_s = 0.0
            M.inflight.inc()
            try:
                # The deadline spans slot acquisition *and* generation. An
                # earlier version wrapped only the generate call, which put the
                # guard on the far side of the queue: during the load incident
                # callers waited 88s for a slot and the 25s deadline never
                # fired, because the clock only started once the wait was over.
                # A deadline that begins after the queueing is not a deadline.
                async with asyncio.timeout(SETTINGS.fallback_deadline_s):
                    async with self._slots:
                        queue_wait_s = time.perf_counter() - t_enqueue
                        M.queue_wait.labels(**labels).observe(queue_wait_s)
                        span.set_attribute(
                            T.A_QUEUE_WAIT_MS, round(queue_wait_s * 1000, 2)
                        )
                        span.add_event(
                            "slot.acquired", {"queue_wait_ms": queue_wait_s * 1000}
                        )

                        t0 = time.perf_counter()
                        _attempt_no.set(0)
                        outcome = await run_in_threadpool(
                            self.generator.generate,
                            model=model,
                            task_id=task_id or request_id,
                            user_text=text,
                            model_cls=model_cls,
                            mode=mode,  # type: ignore[arg-type]
                            temperature=(
                                SETTINGS.default_temperature
                                if temperature is None
                                else temperature
                            ),
                            seed=SETTINGS.default_seed if seed is None else seed,
                            num_predict=num_predict or SETTINGS.default_num_predict,
                        )
                        generate_s = time.perf_counter() - t0
                # Total is measured from arrival, not from slot acquisition.
                # Measuring from t0 was the first version of this line and it
                # was wrong in the way that matters: under load it reported a
                # P95 of 5.7s while callers were waiting 34s, because the queue
                # -- the entire problem -- fell outside the window. A latency
                # metric that starts after the wait cannot alert on the wait.
                # See docs/post_mortem_case_study.md.
                elapsed = time.perf_counter() - t_enqueue
            except TimeoutError:
                # The generation thread is still running -- cancelling the
                # awaitable does not cancel the work behind it. That is the
                # honest trade: the caller stops waiting, the GPU does not stop,
                # and the slot stays held until the model finishes. Reporting
                # the fallback while pretending capacity was freed would make
                # the dashboard lie during exactly the incident it exists for.
                elapsed = time.perf_counter() - t_enqueue
                return self._fallback(
                    span, request_id, trace_id, labels, schema, model, mode,
                    reason="deadline_exceeded", elapsed=elapsed,
                    queue_wait_ms=queue_wait_s * 1000,
                )
            finally:
                M.inflight.dec()

            return self._finalise(
                span, outcome, request_id, trace_id, labels, schema, model, mode,
                elapsed=elapsed, generate_s=generate_s,
                queue_wait_s=queue_wait_s, gold=gold,
            )

    # ------------------------------------------------------------------

    def _finalise(
        self, span, outcome, request_id, trace_id, labels, schema, model, mode,
        *, elapsed: float, generate_s: float, queue_wait_s: float,
        gold: dict[str, Any] | None,
    ) -> ExtractionResult:
        first = outcome.attempts[0] if outcome.attempts else None
        last = outcome.attempts[-1] if outcome.attempts else None

        prompt_tokens = sum(
            (a.gen.prompt_eval_count or 0) for a in outcome.attempts
        ) or None
        completion_tokens = sum(
            (a.gen.eval_count or 0) for a in outcome.attempts
        ) or None
        gpu_seconds = sum(
            (a.gen.total_latency_ms or 0.0) for a in outcome.attempts
        ) / 1000.0

        # Every attempt gets an event carrying the harness's own verdict --
        # the same `FailureKind` taxonomy Project 2's report is built on, so a
        # trace and the benchmark tables can be read against each other.
        for a in outcome.attempts:
            span.add_event(
                f"attempt.{a.attempt}",
                {
                    "ok": a.ok,
                    "failure": a.failure,
                    "error_detail": (a.error_detail or "")[:500],
                    "needed_extraction": a.needed_extraction,
                    "finish_reason": a.gen.finish_reason or "unknown",
                    "latency_ms": a.gen.total_latency_ms or -1.0,
                },
            )
            M.attempts_total.labels(**labels).inc()
            if a.needed_extraction:
                M.extraction_repairs_total.labels(**labels).inc()

        if outcome.n_attempts > 1:
            M.retries_total.labels(**labels).inc(outcome.n_attempts - 1)
            if outcome.recovered_by_retry:
                M.retry_recovered_total.labels(**labels).inc()

        outcome_label = "ok" if outcome.ok else outcome.final_failure
        M.request_duration.labels(**labels, outcome=outcome_label).observe(elapsed)
        M.generation_duration.labels(**labels).observe(generate_s)
        M.requests_total.labels(**labels, outcome=outcome_label).inc()
        if not outcome.ok:
            M.failures_total.labels(**labels, kind=outcome.final_failure).inc()

        if first and first.gen.ttft_ms is not None:
            M.ttft.labels(**labels).observe(first.gen.ttft_ms / 1000.0)
        for a in outcome.attempts:
            if a.gen.eval_duration_ms is not None:
                M.server_eval_duration.labels(**labels).observe(
                    a.gen.eval_duration_ms / 1000.0
                )
        if prompt_tokens:
            M.tokens_total.labels(**labels, direction="prompt").inc(prompt_tokens)
        if completion_tokens:
            M.tokens_total.labels(**labels, direction="completion").inc(completion_tokens)
        if last and last.gen.peak_vram_mib:
            M.vram_peak_mib.labels(model=model).set(last.gen.peak_vram_mib)

        cost = gpu_cost_usd(gpu_seconds)
        hosted = hosted_cost_usd(prompt_tokens, completion_tokens, DEFAULT_HOSTED)
        M.gpu_seconds_total.labels(**labels).inc(gpu_seconds)
        M.cost_usd_total.labels(**labels, basis="gpu").inc(cost)
        M.cost_usd_total.labels(**labels, basis=f"hosted:{DEFAULT_HOSTED}").inc(hosted)

        span.set_attribute(T.A_ATTEMPT, outcome.n_attempts)
        span.set_attribute(T.A_RECOVERED, outcome.recovered_by_retry)
        span.set_attribute(T.A_FAILURE_KIND, outcome.final_failure)
        span.set_attribute(T.A_COST_GPU_S, round(gpu_seconds, 4))
        span.set_attribute(T.A_COST_USD, round(cost, 8))
        span.set_attribute("llmobs.hosted_cost_usd", round(hosted, 8))
        if outcome.parsed is not None:
            T.set_payload(span, T.OUTPUT_VALUE, outcome.parsed, T.OUTPUT_MIME)

        grade = None
        if gold is not None and outcome.parsed is not None:
            grade = self._grade(outcome.parsed, gold, labels, span)

        if not outcome.ok:
            span.set_status(Status(StatusCode.ERROR, outcome.final_failure))
            M.fallbacks_total.labels(**labels, reason=outcome.final_failure).inc()
            span.set_attribute(T.A_FALLBACK, True)

        return ExtractionResult(
            request_id=request_id,
            trace_id=trace_id,
            ok=outcome.ok,
            record=outcome.parsed if outcome.ok else dict(FALLBACK_RECORD),
            schema=schema,
            model=model,
            mode=mode,
            fallback=not outcome.ok,
            failure=outcome.final_failure,
            attempts=outcome.n_attempts,
            recovered_by_retry=outcome.recovered_by_retry,
            latency_ms=round(elapsed * 1000, 2),
            generate_ms=round(generate_s * 1000, 2),
            queue_wait_ms=round(queue_wait_s * 1000, 2),
            ttft_ms=first.gen.ttft_ms if first else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            gpu_seconds=round(gpu_seconds, 4),
            cost_usd=cost,
            hosted_cost_usd=hosted,
            grade=grade,
            warnings=(
                ["response required client-side JSON extraction despite format constraint"]
                if last and last.needed_extraction
                else []
            ),
        )

    def _grade(self, predicted, gold, labels, span) -> dict[str, Any]:
        """Score against gold labels and split errors by direction.

        Commission and omission are counted separately because Project 2 found
        the aggregate hides the dominant failure: models omit far more than
        they invent, and a hallucination-only metric scores that as perfect.
        """
        mean, per_field = score_record(predicted, gold)
        commissions = omissions = 0
        for k, s in per_field.items():
            if s == 1.0:
                M.graded_fields_total.labels(**labels, result="correct").inc()
                continue
            M.graded_fields_total.labels(**labels, result="incorrect").inc()
            g, p = gold.get(k), predicted.get(k)
            if g is None and p is not None:
                commissions += 1
                M.graded_errors_total.labels(**labels, direction="commission").inc()
            elif g is not None and p is None:
                omissions += 1
                M.graded_errors_total.labels(**labels, direction="omission").inc()

        M.field_accuracy.labels(**labels).set(mean)
        span.set_attribute(T.A_FIELD_ACCURACY, round(mean, 4))
        span.set_attribute(T.A_COMMISSIONS, commissions)
        span.set_attribute(T.A_OMISSIONS, omissions)
        return {
            "field_accuracy": round(mean, 4),
            "per_field": per_field,
            "commissions": commissions,
            "omissions": omissions,
        }

    def _fallback(
        self, span, request_id, trace_id, labels, schema, model, mode,
        *, reason: str, elapsed: float, queue_wait_ms: float,
    ) -> ExtractionResult:
        """Degrade deliberately rather than time out.

        A caller that gets a 504 has to guess what happened; a caller that gets
        a labelled abstention can route the record to a human. The distinction
        matters more than the extra second of waiting it costs.
        """
        M.fallbacks_total.labels(**labels, reason=reason).inc()
        M.requests_total.labels(**labels, outcome=reason).inc()
        M.request_duration.labels(**labels, outcome=reason).observe(elapsed)
        span.set_attribute(T.A_FALLBACK, True)
        span.set_attribute(T.A_FAILURE_KIND, reason)
        span.set_status(Status(StatusCode.ERROR, reason))
        return ExtractionResult(
            request_id=request_id, trace_id=trace_id, ok=False,
            record=dict(FALLBACK_RECORD), schema=schema, model=model, mode=mode,
            fallback=True, failure=reason, latency_ms=round(elapsed * 1000, 2),
            queue_wait_ms=round(queue_wait_ms, 2),
        )
