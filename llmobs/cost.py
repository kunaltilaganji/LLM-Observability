"""Cost accounting for local inference, plus a hosted-API counterfactual.

The spec this project follows asks for "real-time API cost per request
calculated down to fractional cents". A locally-served model has no per-token
bill, so that metric as written is identically zero and says nothing. Reporting
$0.00 per request and calling it cost tracking would be a dodge.

Two things are measured instead:

1. **What this request actually cost to serve.** Local inference is paid for in
   GPU-time, so the unit cost is GPU-seconds x the hourly rate for the hardware.
   That is the number an operator uses to decide whether to keep serving.

2. **What the same request would have cost on a hosted API.** Token counts are
   already measured, so the counterfactual is a multiplication -- and it is the
   number that actually justifies the architecture. "Self-hosting is cheaper"
   is an assertion until both sides of it are metered.

Both are exported as Prometheus counters so the dashboard can show the crossover
point: below some request volume the hosted API is cheaper, because a GPU bills
for wall-clock time whether or not anything is running.
"""

from __future__ import annotations

from dataclasses import dataclass

from .settings import SETTINGS


@dataclass(frozen=True)
class HostedRate:
    """Published per-million-token pricing for a hosted model."""

    model_id: str
    label: str
    input_per_mtok_usd: float
    output_per_mtok_usd: float
    note: str = ""


# Anthropic first-party API list pricing, as published on 2026-08-08.
# Haiku 4.5 is the honest comparison for this workload -- structured extraction
# from short documents is exactly the "simple, speed-critical" task it targets,
# so pricing against a frontier model would flatter the local deployment.
# Sonnet 5 is included as the upper bound for a caller who wants the accuracy
# headroom. Verify before quoting these in a report; vendor pricing moves.
HOSTED_RATES: dict[str, HostedRate] = {
    "haiku": HostedRate(
        model_id="claude-haiku-4-5",
        label="Claude Haiku 4.5",
        input_per_mtok_usd=1.00,
        output_per_mtok_usd=5.00,
        note="cheapest current Claude model; the like-for-like comparison",
    ),
    "sonnet": HostedRate(
        model_id="claude-sonnet-5",
        label="Claude Sonnet 5",
        input_per_mtok_usd=2.00,
        output_per_mtok_usd=10.00,
        note="introductory rate through 2026-08-31; list is $3.00/$15.00",
    ),
}

DEFAULT_HOSTED = "haiku"


def gpu_cost_usd(gpu_seconds: float, hourly_usd: float | None = None) -> float:
    """Amortised hardware cost of holding the GPU for this request."""
    rate = SETTINGS.gpu_hourly_usd if hourly_usd is None else hourly_usd
    return gpu_seconds * (rate / 3600.0)


def hosted_cost_usd(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    rate_key: str = DEFAULT_HOSTED,
) -> float:
    """What this same request would have cost on a hosted API.

    Token counts come from Ollama's own accounting, so this compares the local
    tokenizer's counts against another vendor's prices -- close enough to size
    a decision, not close enough to reconcile an invoice. The report says so.
    """
    rate = HOSTED_RATES.get(rate_key) or HOSTED_RATES[DEFAULT_HOSTED]
    p = (prompt_tokens or 0) / 1_000_000 * rate.input_per_mtok_usd
    c = (completion_tokens or 0) / 1_000_000 * rate.output_per_mtok_usd
    return p + c


def breakeven_requests_per_hour(
    mean_gpu_seconds: float,
    mean_prompt_tokens: float,
    mean_completion_tokens: float,
    rate_key: str = DEFAULT_HOSTED,
) -> float | None:
    """Requests/hour at which the GPU stops being the more expensive option.

    A rented GPU bills for the whole hour whether it serves one request or ten
    thousand; a hosted API bills only per token. So the local deployment wins
    above a volume threshold and loses below it, and quoting a single
    cost-per-request without that threshold is how self-hosting gets oversold.

    Returns None when the hosted cost per request is zero (nothing to cross).
    """
    hosted_per_request = hosted_cost_usd(
        int(mean_prompt_tokens), int(mean_completion_tokens), rate_key
    )
    if hosted_per_request <= 0:
        return None
    # Idle GPU-hours are the fixed cost; serving is assumed to fit within the
    # hour rather than adding to it, which is the charitable reading for local.
    del mean_gpu_seconds
    return SETTINGS.gpu_hourly_usd / hosted_per_request
