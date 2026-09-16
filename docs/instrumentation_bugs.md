# Instrumentation bugs found in this project

Four bugs, all found by running a real load incident against the service rather
than by reading the code. Every one of them was in the *measurement* layer, not
the serving layer — the service worked correctly throughout; the instrumentation
lied about what it was doing.

They share a single root cause, which is the thing worth being able to state in
one sentence:

> **Every one of these bugs is a timer, guard, or bucket scoped to the wrong
> interval. The code was correct about *what* it measured and wrong about
> *when* it started and stopped measuring.**

That is why they all survived code review by inspection — each line is
individually reasonable — and why they all died instantly under load. A latency
metric with the wrong window looks perfect at 1 request/second and is worthless
at 4.

Numbers below are measured, from `reports/incident_*.json` and Prometheus
snapshots taken during the runs. Hardware: RTX A6000 (GPU 2), `gemma3:4b`,
`schema` enforcement mode, `max_concurrency=4`.

---

## Bug 1 — the latency metric excluded the queue it claimed to measure

### What the code said

`llmobs_request_duration_seconds` was documented as *"End-to-end wall-clock
latency of /v1/extract, including retries and queueing."*

### What the code did

```python
async with self._slots:                       # <-- queue wait happens HERE
    queue_wait_s = time.perf_counter() - t_enqueue
    t0 = time.perf_counter()                  # <-- timer starts AFTER the wait
    outcome = await asyncio.wait_for(run_in_threadpool(...), timeout=...)
    elapsed = time.perf_counter() - t0        # <-- excludes the queue entirely
```

`t0` is taken *after* `async with self._slots` has already blocked. Every second
spent waiting for a concurrency slot falls outside the measured window.

### The evidence

During a 4 req/s spike against a service sized for roughly 0.75 req/s:

| measurement | value |
|---|---:|
| what the service reported as request P95 | **5.65 s** |
| semaphore queue wait P95 (measured separately) | **28.9 s** |
| what the caller actually waited (client-side P95) | **352 s** |

The reported P95 was smaller than a single component of itself. That
impossibility is the tell — `request_duration` was not a total, it was a
sub-interval mislabelled as a total.

### Why it mattered

The SLO alert reads the broken metric:

```yaml
- alert: ExtractionLatencyP95Breach
  expr: llmobs:request_latency_seconds:p95 > 8
```

5.65 < 8, so **the latency alert never fired** during a latency incident.
Three other alerts did fire (`QueueWaitDominatingLatency`,
`GradedAccuracyRegression`, `OmissionRateSpike`) — so the dashboard was not
silent, it was *misleading*: it showed a queueing warning next to a latency
panel that looked healthy, which invites exactly the wrong diagnosis ("queue is
backing up but responses are still fast, so we're fine").

After the fix, the same load fires it immediately:
`ExtractionLatencyP95Breach | P95 latency 60.0s exceeds the 8s SLO`.

### The fix

```python
elapsed = time.perf_counter() - t_enqueue     # from arrival, not from admission
```

and a second metric, `llmobs_generation_duration_seconds`, kept deliberately
separate so the two can be subtracted:

```
request_duration - generation_duration = queueing
```

Keeping both is the actual design point. One number cannot tell you whether a
latency regression came from the model or from the queue; two numbers can, and
the attribution is a subtraction rather than a trace-opening exercise.

### The general lesson

**A latency metric must start when the request arrives, not when you start
working on it.** Anything you measure from the moment you begin serving is a
service-time metric, and service time is not what users experience — response
time is (queue wait + service time). This is textbook queueing theory
(response time = wait + service) and it is extremely easy to violate in async
code, because the `await` that blocks on admission looks like ordinary control
flow rather than like a queue.

---

## Bug 2 — the timeout guard was placed on the far side of the queue it guarded

### What the code was for

`fallback_deadline_s = 25.0` — a deadline after which the service stops waiting
and returns an explicit degraded response (`extraction_unavailable`) instead of
making the caller hang. The whole point is load shedding: under overload,
answer *something* fast rather than everything slowly.

### What the code did

```python
async with self._slots:                                    # unbounded wait
    outcome = await asyncio.wait_for(
        run_in_threadpool(self.generator.generate, ...),
        timeout=SETTINGS.fallback_deadline_s,               # 25s clock starts here
    )
```

`asyncio.wait_for` wraps only the generation call. The clock starts *after* the
semaphore has been acquired. Waiting for the semaphore is unbounded.

### The evidence

During the same 4 req/s spike, across 508 requests:

| measurement | value |
|---|---:|
| slot wait, mean | **88.5 s** |
| slot wait, max | **175.4 s** |
| generation time once admitted, mean | 3.8 s |
| `deadline_exceeded` fallbacks fired | **0** |
| client-side `ReadTimeout` after 180 s | **258** |

Zero deadlines fired while 258 callers gave up. The guard was structurally
incapable of firing: generation itself never took 25 s, so the only thing that
could trip the deadline was the wait — and the wait was outside the guard.

### Why it mattered

This is worse than bug 1. Bug 1 made the dashboard misreport an incident; bug 2
meant the **mitigation for that incident did not exist**. The service had a
load-shedding mechanism, a config value for it, a metric for it
(`llmobs_fallbacks_total{reason="deadline_exceeded"}`), an alert on it — and
none of it could ever engage, because the timer was nested inside the thing it
was supposed to time out.

### The fix

```python
async with asyncio.timeout(SETTINGS.fallback_deadline_s):   # covers BOTH
    async with self._slots:                                 # admission
        outcome = await run_in_threadpool(...)              # and generation
```

Verified working: a run where Ollama was still draining a prior backlog produced
16 `deadline_exceeded` fallbacks out of 51 requests — the guard shedding load
exactly as designed, which it had never once done before the fix.

### An honest caveat kept in the code

Cancelling the awaitable does **not** cancel the generation running in the
thread pool. The caller stops waiting; the GPU does not stop working, and the
slot stays held until the model finishes. The comment in `pipeline.py` says so
explicitly, because a fallback that claimed to free capacity would make the
dashboard lie during exactly the incident it exists for. This is a real
limitation, not a solved problem — proper fixes are cooperative cancellation in
the client or admission control that rejects before queueing.

### The general lesson

**A deadline must span every phase the caller waits through, including the
phases where you are doing nothing.** Timeouts get scoped to "the expensive
operation" because that is what feels risky, but queueing — where the service
is idle from its own point of view — is usually where overload latency actually
accumulates.

---

## Bug 3 — histogram buckets saturated, so the metric flatlined at its peak

### What happened

Prometheus reported, during the spike:

| metric | reported |
|---|---:|
| `request_duration` P95 | **60.0 s** |
| `queue_wait` P95 | **30.0 s** |

Both values are *exactly* the top finite bucket boundary of their respective
histograms:

```python
_LATENCY_BUCKETS = (..., 20.0, 30.0, 60.0)      # top finite bound = 60
_QUEUE_BUCKETS   = (..., 5.0, 10.0, 30.0)       # top finite bound = 30
```

Raw client-side data from the same run showed the true server-side P95 was
**170.4 s**.

### Why it happens

`histogram_quantile()` interpolates *within* buckets. Once observations land in
the `+Inf` bucket, there is no upper bound to interpolate toward, so Prometheus
returns the highest finite bound. The metric does not error, does not go stale,
and does not flag itself — it silently pins to a plausible-looking number and
stays there no matter how bad things get.

A P95 sitting exactly on a round bucket boundary is therefore a saturated
metric, not a measurement. `60.0` and `30.0` with a trailing `.0` should be read
as "at least this", never as "this".

### The extra sting

I had explicitly justified these bucket bounds in a code comment — *"chosen from
Project 2's measured latency distribution on this task, so the interesting part
of the curve is where the resolution is"* — which is exactly right for steady
state and exactly wrong for an incident. Buckets tuned to normal operation blind
you precisely when you need the number most, because incidents are by definition
outside the normal distribution.

### The fix

```python
_LATENCY_BUCKETS = (..., 30.0, 60.0, 120.0, 300.0)
_QUEUE_BUCKETS   = (..., 30.0, 60.0, 120.0, 300.0)
```

### The general lesson

**Histogram buckets bound what is observable, not just what is precise.** Size
the top bucket for the incident you are trying to catch, not for the steady
state you expect — the cost of a couple of extra buckets is a few time series;
the cost of saturating is a blind dashboard during an outage. And when a
quantile lands on a bucket boundary, check saturation before believing it.

---

## Bug 4 — the load generator was the bottleneck, so the first run measured the wrong system

### What happened

The first incident run reported a client-side P95 of **352 s** while the
service's own instrumentation (once bug 1 was accounted for) put the server-side
total at roughly **34 s**. Around 318 s — over 90 % of what the caller waited —
was accumulating somewhere neither side could see.

Prometheus gave it away:

```
peak in-flight requests: 32.0
```

Exactly 32, exactly flat. Real concurrency does not land on a round number and
stay there. That was a limit, and it was mine:

```python
limits = httpx.Limits(max_connections=max(args.rate * 8, 32))   # 4*8 = 32
```

The load generator's own connection pool admitted 32 concurrent requests. The
remaining ~450 requests queued *inside the client*, before a socket was ever
opened. The run measured `loadgen.py`, not the service.

### Why it mattered

This one produced a *correct-looking* incident with a completely wrong cause. A
352 s P95 with the service reporting ~34 s invites the conclusion "the network
or the framework is the problem", when in fact the test harness was throttling
itself. Had the fix for bug 1 landed first, the discrepancy would have been
mistaken for evidence of a server-side problem that did not exist.

### The fix

Two parts. First, stop self-throttling:

```python
p.add_argument("--max-connections", type=int, default=512,
               help="keep well above the in-flight count or the generator "
                    "throttles itself")
```

Second — the more useful half — record **both** latencies on every request and
print the gap:

```python
"latency_s":         elapsed,            # what this client waited
"server_latency_ms": body["latency_ms"], # what the service believes it took
```

```
** 9.6s of the P95 is invisible to the service (5% of what the caller waited) **
```

The gap between client-observed and server-observed latency is now a
first-class output, so anything living outside the service — connection pools,
socket backlog, the network, or the load generator itself — shows up as a
number instead of as a mystery.

### The general lesson

**Always instrument the load generator too, and always record client- and
server-observed latency separately.** A load test that records only one number
cannot distinguish "the service is slow" from "my test harness is slow", and
those have opposite fixes. Round, flat concurrency ceilings are the diagnostic
signature — real systems produce noisy numbers.

---

## How these connect to Project 2

This project is the serving and observability layer over
[`offline-slm-assistant`](../../offline-slm-assistant), whose own harness bugs
are written up in that repo's `docs/instrumentation_bugs.md`. The theme carries
across both:

| Project 2 | Project 3 |
|---|---|
| a one-sided quality metric hid a 30 % failure rate | a wrongly-scoped latency metric hid a 60× latency breach |
| gold labels were underspecified, so model behaviour was scored against an undefined rubric | histogram buckets were undersized, so latency was scored against an unreachable ceiling |

Both reduce to the same discipline: **the measurement apparatus is part of the
system under test, and it fails in ways that look like success.** Project 2
found that by getting a suspiciously perfect hallucination rate. Project 3 found
it by getting a suspiciously calm latency panel during an outage.

---

## The 60-second version, for when someone asks

> I built a monitoring layer for a local LLM serving stack, then ran a real
> overload incident against it — and the incident found four bugs in my own
> instrumentation before it found anything about the service.
>
> The latency metric started its timer after the request was admitted, so it
> excluded the queue and the SLO alert stayed silent while callers waited 352
> seconds. The load-shedding timeout was nested inside that same queue, so it
> could never fire — 258 clients timed out and zero fallbacks triggered. The
> histogram's top bucket was 60 seconds, so the P95 pinned at exactly 60.0 while
> the real value was 170. And the first run was invalid anyway, because my load
> generator's connection pool capped concurrency at 32 and I was measuring the
> test harness.
>
> They're all the same bug wearing different clothes: a timer, a guard, and a
> bucket each scoped to the wrong interval. Every one looked fine in review and
> at one request per second. That's why I now record client- and server-observed
> latency separately on every request — the gap between them is where the
> problems you can't see from inside the service live.
