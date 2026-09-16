# Post-mortem: a latency incident the server-side metric could not see

**Service:** `llm-extraction-service` (local `gemma3:4b`, single RTX A6000)
**Date:** 2026-08-08 · **Host:** `aicoe-a6000` · **Author:** Kunal
**Severity:** SEV2 — sustained SLO breach, no data loss
**Status:** resolved; three defects fixed, one accepted

Every number below comes from `reports/incident_timeline.json`, produced by
`scripts/incident.py`. The incident is reproducible with one command.

---

## Summary

A 8× traffic increase (0.5 → 4 req/s) against a single-GPU deployment drove the
P95 latency from 2.8s to 25s and pushed 83% of requests onto the degraded
fallback path. No component failed. No error was logged by the model server.

**Ollama's own reported decode time did not move at all:**

| Phase | Offered load | Client P95 | Service P95 | Queue wait P95 | Generation P95 | **Ollama-reported decode P95** | Peak in-flight | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — steady, before | 0.5 req/s | 2.82s | 3.50s | 0.00s | 3.50s | **0.975s** | 4 | 0% |
| B — spike | 4 req/s | 25.02s | 29.41s | 28.86s | 3.91s | **0.973s** | 109 | 83% |
| C — steady, after | 0.5 req/s | 2.71s | 2.81s | 0.00s | 2.81s | **0.973s** | 3 | 0% |

The model server's own latency metric is flat to three decimal places across an
incident where user-observed latency rose roughly tenfold. A dashboard built on
`eval_duration` — the number the inference server volunteers, and the one most
readily available — would have drawn a horizontal line through the entire
event.

This is the finding the project exists to demonstrate. It is not a bug in
Ollama: `eval_duration` measures decode and is accurate. It is a bug in
*choosing* it as the latency SLI. **The server can only report time it spent
working. It cannot report time you spent waiting for it.**

---

## Timeline

| Time | Event |
|---|---|
| T+0 | Phase A. 0.5 req/s. P95 2.82s, queue wait 0ms, in-flight ≤ 4. All SLOs met. |
| T+95s | Offered load raised to 4 req/s. |
| T+~100s | In-flight climbs past `max_concurrency` (4); requests begin queueing. |
| T+~110s | `QueueWaitDominatingLatency` fires (warn). Queue wait P95 > 2s. |
| T+~130s | `ExtractionLatencyP95Breach` fires (page). P95 > 8s SLO. |
| T+~140s | Deadline guard begins shedding: requests exceeding 25s return the abstention. Fallback rate reaches 83%. |
| T+185s | Load generator stops. **Backlog continues draining for ~4 minutes** — cancelling a request does not cancel the generation behind it. |
| T+~430s | Ollama probe returns < 1s; backlog clear. |
| T+~440s | Phase C. 0.5 req/s. P95 2.71s. Latency alerts clear. |

---

## Root cause

One GPU serving one model is a single-server queue. At 0.5 req/s with a mean
service time of ~1.7s the utilisation is ~85% of one server — already close to
the knee. At 4 req/s the arrival rate exceeds the service rate outright, so the
queue does not reach a steady state; it grows for as long as the load lasts.
Nothing is broken. The system is doing exactly what an over-subscribed queue
does.

The interesting part is the decomposition. During phase B:

```
service P95            29.41s
  ├── queue wait       28.86s   ← 98% of it
  └── generation        3.91s
        └── Ollama-reported decode   0.97s
```

Generation itself did rise — 3.50s → 3.91s — because four concurrent requests
share one GPU. But that is a 12% change inside a 10× incident. **98% of the
latency was time spent waiting for a slot**, which is invisible to any metric
emitted by the thing being waited for.

---

## What went wrong in the instrumentation

Three defects were found while running this, and finding them is most of what
the exercise was worth. All three are the same shape: **the instrument was
placed on the wrong side of the thing it was supposed to measure.**

### 1. The latency histogram excluded the queue

The first version started the timer after acquiring the concurrency slot:

```python
async with self._slots:          # ← wait happens here, untimed
    t0 = time.perf_counter()
    outcome = await generate(...)
    elapsed = time.perf_counter() - t0    # ← wrong
```

During the first spike run this reported a P95 of **5.65s** while callers were
waiting **170s**. `ExtractionLatencyP95Breach` never fired. The SLO was
comfortably met by a metric that had been defined to exclude the entire
problem.

**Fix:** measure from arrival (`llmobs/pipeline.py`). `request_duration` now
starts at `t_enqueue`; `generation_duration` was added alongside it so the two
can be subtracted. If generation moved, it is the model; if only the total
moved, it is the queue — attributable without opening a trace.

### 2. The deadline guard was inside the queue

`fallback_deadline_s = 25` was meant to shed load rather than let callers wait
indefinitely. It was implemented as `asyncio.wait_for` around the generate call
— which sits *after* slot acquisition. During the first spike, callers waited
88s for a slot and the 25s deadline never fired once, because its clock only
started when the wait was already over. 508 requests, zero deadline
fallbacks; the load generator's own 180s timeout was the only thing bounding
anything.

**Fix:** `asyncio.timeout(...)` now wraps slot acquisition *and* generation. In
the clean run the guard did its job: 338 of 372 requests shed at 25s, and the
service stayed responsive instead of accumulating an unbounded backlog. A
deadline that begins after the queueing is not a deadline.

### 3. The histogram buckets were too narrow to see the peak

Buckets topped out at 60s, chosen from Project 2's measured distribution and
justified in a code comment as being where the interesting part of the curve
is. `histogram_quantile` cannot return a value above the highest finite bound,
so during the first spike the dashboard reported a P95 of exactly **60.0s**
while real latencies ran to 170s — and the queue-wait panel reported exactly
**30.0s**, its own top bucket.

A percentile sitting precisely on a bucket boundary is a saturated metric, not
a measurement. **Fix:** buckets extended to 120s and 300s. The lesson
generalises: histogram buckets bound what is observable, and an incident that
exceeds them is invisible exactly when it is worst.

---

## What the alerts got right and wrong

**Right.** `QueueWaitDominatingLatency` fired before the latency SLO breach and
named the cause directly, and its annotation pointed at the correct next step
("the levers are concurrency, admission control, or hardware — raising
num_predict or switching model will make this worse"). Diagnosis took one panel.

**Wrong.** `GradedAccuracyRegression` and `OmissionRateSpike` fired in *all
three phases*, including both healthy ones. They are not incident signal; they
are permanently-firing noise, and this run is what exposed them.

The cause is a threshold set from an aggregate but evaluated per-label-set. The
recorded baseline is 77.5% field accuracy across all 52 tasks, so the floor was
set at 75%. But the alert groups by `(model, schema, mode)`, and the per-schema
split straddles that floor — `action_item` sits near 68%, below the threshold
at all times, while the other schemas sit above it. The alert had been firing
since before the incident and would have kept firing after.

An alert that is always firing is worse than no alert: it trains the on-call to
ignore the channel, and during phase B it contributed two of the six firing
alerts with no diagnostic value. **Action:** set per-schema thresholds from the
per-schema baselines (`eval/baseline.json` already records `by_tag`), or gate
on a drop relative to baseline rather than an absolute floor. Tracked below;
not fixed in this pass, and it should be fixed before this alert is trusted.

---

## Detection

Detected by alert, not by a user report — `QueueWaitDominatingLatency` fired
roughly 15s after the load increase. **After** the metric fix; before it, the
latency SLO alert could not fire at all.

---

## Resolution

Load returned to 0.5 req/s; latency recovered to 2.71s P95 within one scrape
interval of the backlog clearing. No restart, no config change, no data loss:
every shed request returned an explicit `extraction_unavailable` abstention
rather than a partial or guessed record, so no caller received a wrong answer.

The ~4 minute drain after the load stopped is the honest cost of the design.
Cancelling the awaitable does not cancel the generation behind it, so the GPU
keeps working through the backlog after callers have given up. `scripts/incident.py`
waits for a drain probe between phases for exactly this reason — starting the
next phase early measures the previous one, which is how the first attempt at
phase A produced a nonsensical 19s "steady state".

---

## Action items

| # | Action | Status |
|---|---|---|
| 1 | Measure request latency from arrival, not slot acquisition | **Done** |
| 2 | Move the fallback deadline outside the concurrency semaphore | **Done** |
| 3 | Extend latency/queue buckets to 120s and 300s | **Done** |
| 4 | Add `generation_duration` so queue vs. model is attributable from metrics alone | **Done** |
| 5 | Per-schema quality thresholds; stop the always-firing accuracy alert | Open |
| 6 | Admission control — reject at the door above a queue-depth watermark rather than admitting and timing out | Open |
| 7 | Alert on `queue_wait / request_duration` ratio, which distinguishes "slow model" from "too much traffic" in one signal | Open |

---

## What generalises

1. **Never take the inference server's own latency number as your SLI.** It
   measures the work, not the wait. Wall-clock from arrival is the only number
   a user experiences. This is the same trap Project 2 documented for
   throughput benchmarking, one layer up the stack — and it is more dangerous
   here, because in a benchmark a flattering number is embarrassing, while in
   production it is an alert that does not fire.

2. **Measure the queue you own, and know about the ones you do not.** The
   semaphore wait was instrumented and became the diagnosis. Two other queues
   were not: Ollama's internal queue (visible only as inflated generation time)
   and, in the first run, the load generator's own connection pool — which
   capped in-flight at exactly 32 and made the *client* the bottleneck. That
   run measured the test harness, not the service.

3. **Check where your guards sit relative to what they guard.** Both defect 1
   and defect 2 are the same mistake: an instrument placed after a wait cannot
   observe or bound that wait. It is worth auditing every timeout and timer for
   what it actually encloses.

4. **A percentile pinned to a round number is a saturated bucket.** Exactly
   `60.0` and exactly `30.0` were not measurements.

5. **An always-firing alert is a broken alert.** Deriving a per-group threshold
   from an aggregate baseline guarantees the below-average groups fire forever.

---

## Reproducing

```bash
./scripts/stack.sh up
python scripts/incident.py          # ~12 min including drains
```

Writes `reports/incident_timeline.json` — the source of every number above.
