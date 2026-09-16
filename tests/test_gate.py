"""Tests for the parts of the gate that must not silently pass.

The gate's whole value is failing when it should. A gate with a bug that makes
it always pass is worse than no gate, because it is also a green check mark --
so the cases tested here are the ones where a subtle error produces a false
negative rather than a crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.gate import compare, score  # noqa: E402
from llmobs.cost import breakeven_requests_per_hour, hosted_cost_usd  # noqa: E402
from llmobs.slm_bridge import ALL_TASKS, SCHEMAS, score_record  # noqa: E402


# --------------------------------------------------------------------------
# The gold suite itself
# --------------------------------------------------------------------------

def test_every_gold_label_satisfies_its_own_schema():
    """A gold label that fails its own validator would score every model at
    zero on that task and look like a model regression."""
    for task in ALL_TASKS:
        model_cls = SCHEMAS[task.schema]
        payload = dict(task.gold)
        # `title` is excluded from scoring but required by the schema; supply a
        # placeholder so validation exercises the scored fields.
        for name, f in model_cls.model_fields.items():
            if name not in payload and f.is_required():
                payload[name] = "placeholder"
        model_cls.model_validate(payload)


def test_a_perfect_prediction_scores_one():
    for task in ALL_TASKS:
        mean, _ = score_record(dict(task.gold), task.gold)
        assert mean == 1.0, f"{task.id} does not score 1.0 against itself"


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _row(task_id: str, ok: bool, record: dict, tags=()) -> dict:
    return {"task_id": task_id, "schema": "action_item", "tags": list(tags),
            "ok": ok, "failure": "none" if ok else "json_parse_error",
            "record": record, "latency_s": 1.0, "attempts": 1,
            "model": "m", "mode": "schema"}


def test_failed_requests_score_zero_rather_than_being_dropped():
    """Excluding failures would let a model that refuses half the suite post a
    higher accuracy than one that answers all of it."""
    task = ALL_TASKS[0]
    answered = score([_row(task.id, True, dict(task.gold))])
    refused = score([_row(task.id, True, dict(task.gold)),
                     _row(task.id, False, None)])
    assert answered["field_accuracy"] == 1.0
    assert refused["field_accuracy"] == 0.5, "failures must drag the mean down"
    assert refused["schema_validity"] == 0.5


def test_omission_and_commission_are_counted_separately():
    task = next(t for t in ALL_TASKS if any(v is None for v in t.gold.values())
                and any(v is not None for v in t.gold.values()))
    # Drop every present value -> omissions, no commissions.
    dropped = {k: None for k in task.gold}
    s = score([_row(task.id, True, dropped)])
    assert s["omission_rate"] > 0
    assert s["commission_rate"] == 0

    # Invent a value where gold is null -> commission.
    null_field = next(k for k, v in task.gold.items() if v is None)
    invented = dict(task.gold)
    invented[null_field] = "Fabricated"
    s2 = score([_row(task.id, True, invented)])
    assert s2["commission_rate"] > 0
    assert s2["omission_rate"] == 0


# --------------------------------------------------------------------------
# Gate arithmetic -- the false-negative cases
# --------------------------------------------------------------------------

BASELINE = {"field_accuracy": 0.80, "schema_validity": 1.0,
            "latency_p95_s": 2.0, "by_tag": {"null_trap": 0.90}}


def _current(**over) -> dict:
    base = {"field_accuracy": 0.80, "schema_validity": 1.0,
            "latency_p95_s": 2.0, "by_tag": {"null_trap": 0.90}}
    base.update(over)
    return base


def test_gate_passes_when_nothing_moved():
    assert compare(_current(), BASELINE, tolerance=0.03, check_latency=True) == []


def test_gate_fails_on_accuracy_drop_beyond_tolerance():
    v = compare(_current(field_accuracy=0.74), BASELINE, 0.03, True)
    assert any("field accuracy" in x for x in v)


def test_gate_tolerates_a_drop_inside_tolerance():
    assert compare(_current(field_accuracy=0.78), BASELINE, 0.03, True) == []


def test_gate_catches_a_single_failure_mode_collapsing():
    """The aggregate can stay inside tolerance while one failure mode dies --
    this is the case a mean-only gate misses."""
    v = compare(_current(by_tag={"null_trap": 0.40}), BASELINE, 0.03, True)
    assert any("null_trap" in x for x in v)


def test_latency_is_not_gated_in_replay():
    slow = _current(latency_p95_s=999.0)
    assert compare(slow, BASELINE, 0.03, check_latency=False) == []
    assert compare(slow, BASELINE, 0.03, check_latency=True) != []


def test_absolute_floor_applies_even_if_the_baseline_is_low():
    """A baseline recorded from an already-bad run must not license staying bad."""
    weak = {"field_accuracy": 0.50, "schema_validity": 1.0,
            "latency_p95_s": 1.0, "by_tag": {}}
    v = compare(_current(field_accuracy=0.50), weak, 0.03, True)
    assert any("SLO floor" in x for x in v)


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------

def test_hosted_cost_uses_separate_input_and_output_rates():
    # Output tokens are priced higher than input on every hosted model here,
    # so swapping the two must not produce the same number.
    a = hosted_cost_usd(1_000_000, 0, "haiku")
    b = hosted_cost_usd(0, 1_000_000, "haiku")
    assert a == pytest.approx(1.00)
    assert b == pytest.approx(5.00)


def test_breakeven_is_none_when_there_is_nothing_to_cross():
    assert breakeven_requests_per_hour(1.0, 0, 0) is None


def test_breakeven_falls_as_requests_get_more_expensive():
    cheap = breakeven_requests_per_hour(1.0, 100, 20)
    dear = breakeven_requests_per_hour(1.0, 10_000, 2_000)
    assert cheap > dear, "pricier requests should break even at lower volume"
