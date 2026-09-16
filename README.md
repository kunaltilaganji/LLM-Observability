# LLM Observability — instrumenting a local extraction service until it could be debugged

A structured-extraction API served from a local model on one GPU, wrapped in
OpenTelemetry tracing, Prometheus metrics, Grafana dashboards, and a regression
gate — then deliberately broken under load to find out what the instrumentation
could and could not see.

The deliverable is not the dashboard. It is
[`docs/post_mortem_case_study.md`](docs/post_mortem_case_study.md): a real
incident, diagnosed from these signals, including the three defects the
instrumentation had in itself.

This is the serving-and-operations half of
[`offline-slm-assistant`](../offline-slm-assistant), which measured *which*
small model to run. This one measures what happens once it is running.

---

## The finding

An 8× traffic increase against a single-GPU deployment drove user-observed P95
latency from 2.8s to 25s and shed 83% of requests onto a degraded fallback.

Ollama's own reported decode time did not move:

| Phase | Load | Client P95 | Service P95 | Queue wait P95 | **Ollama-reported decode P95** |
|---|---:|---:|---:|---:|---:|
| steady, before | 0.5 req/s | 2.82s | 3.50s | 0.00s | **0.975s** |
| **spike** | **4 req/s** | **25.02s** | **29.41s** | **28.86s** | **0.973s** |
| steady, after | 0.5 req/s | 2.71s | 2.81s | 0.00s | **0.973s** |

Flat to three decimal places across a tenfold latency incident. A dashboard
built on the metric the inference server volunteers — which is the easiest one
to reach for — would have drawn a horizontal line through the whole event.

That is not a bug in Ollama. `eval_duration` measures decode, and it measures it
correctly. It is a bug in choosing it as the SLI. **A server can only report
time it spent working; it cannot report time you spent waiting for it.**

98% of the incident's latency was queueing.

---

## Why this project exists

Most observability write-ups end at "I installed a tracing library and here is a
screenshot." That proves the library installs. It does not show that the signals
are the *right* signals, which is only testable by breaking something and seeing
whether the dashboard says so.

So the deliberate structure here is: instrument, break, diagnose, and report
what the instrumentation got wrong. Three of its own defects turned up that way,
and all three are the same shape — **an instrument placed on the wrong side of
the thing it was supposed to measure**:

1. **The latency histogram excluded the queue.** The timer started after
   acquiring the concurrency slot. It reported a P95 of 5.65s while callers
   waited 170s, so the SLO alert never fired. The metric had been defined to
   exclude the entire problem.

2. **The load-shedding deadline sat inside the queue it was guarding.** A 25s
   deadline wrapped only the generate call, which runs *after* slot
   acquisition. Callers waited 88s and it fired zero times across 508 requests.
   A deadline whose clock starts after the wait is not a deadline.

3. **The histogram buckets were too narrow to see the peak.** Top finite bucket
   was 60s, so `histogram_quantile` reported exactly `60.0` while real
   latencies ran to 170s. A percentile pinned to a round number is a saturated
   bucket, not a measurement.

A fourth was in the load generator, not the service: its connection pool capped
in-flight requests at exactly 32, so the first run measured the test harness
rather than the service.

Full write-ups: [`docs/post_mortem_case_study.md`](docs/post_mortem_case_study.md).

---

## Architecture

```
client
  │
  ▼
FastAPI  /v1/extract ──────────────────────────────────► Prometheus ──► Grafana
  │  ├─ concurrency semaphore   (queue wait measured)        ▲             │
  │  ├─ deadline guard          (sheds load, abstains)       │        13 panels,
  │  └─ StructuredGenerator ────────────────► Ollama    /metrics      9 alert rules
  │        (imported from Project 2:          (gemma3:4b,
  │         schemas, retry loop, scorers)      RTX A6000)
  │
  └─ OpenTelemetry spans ──► OTLP ──► Phoenix        (trace UI)
                          └► JSONL ──► results/traces  (durable evidence)
```

**Project 2 is imported, not forked.** The schemas, the corrective retry loop,
the field-level scorers, and the 52 gold-labelled tasks all come from the
sibling repo via [`llmobs/slm_bridge.py`](llmobs/slm_bridge.py). Copying them
would let the two drift and quietly invalidate Project 2's published report —
and the regression gate's whole value is that it grades against *the same*
labels. The coupling is deliberately one file wide.

The tracing follows from that. Rather than reimplement the retry loop to get
per-attempt spans, a traced subclass of the Ollama client opens a span around
every model call, and Project 2's loop nests inside it without knowing
OpenTelemetry exists. Parse and validation happen inside that loop where no span
can be opened without forking it, so they are recorded as span *events* carrying
the harness's own failure taxonomy. **Events describing real outcomes beat spans
with invented timestamps.**

---

## What is measured

**SRE.** P50/P90/P95 for end-to-end latency, TTFT, queue wait, and generation
time — plus Ollama's server-reported decode, kept deliberately alongside so the
gap between felt and admitted latency is a panel rather than an investigation.

**Reliability.** Schema validity, retry rate, retry-recovery rate, fallback
rate, failures split by Project 2's taxonomy (`truncated`, `json_parse_error`,
`transport_error`, …), and the rate at which *constrained* output still needed
client-side JSON repair — which Project 2 found is not zero even in schema mode.

**Quality, on graded traffic only.** Field accuracy against gold labels, scored
programmatically, split into **omission** (dropped a real value) and
**commission** (invented one). Only requests carrying a `task_id` are scored;
organic traffic has no labels and is deliberately not counted. Online quality
metrics require labels, and reporting accuracy for unlabelled traffic would be
inventing a number.

**Cost.** The spec this follows asks for API cost per request. A local model has
no per-token bill, so that metric is identically zero and says nothing. Two real
numbers replace it:

- **What serving actually cost** — GPU-seconds × hourly rate, the number that
  decides whether to keep the hardware.
- **What the same tokens would have cost hosted** — priced against
  `claude-haiku-4-5` at $1.00/$5.00 per Mtok, the like-for-like comparison for
  short-document extraction.

Both are exported, so the dashboard shows the crossover: a GPU bills for the
whole hour whether or not anything is running, so self-hosting wins only above a
volume threshold. Quoting a cost-per-request without that threshold is how
self-hosting gets oversold.

---

## The regression gate

[`eval/gate.py`](eval/gate.py) grades the live service against Project 2's 52
gold tasks and fails the build on:

- field accuracy dropping more than 3 points below the recorded baseline
- field accuracy below an absolute floor, regardless of baseline (so a baseline
  recorded from an already-bad run cannot license staying bad)
- schema validity below 98%
- P95 latency above the SLO (live runs only)
- **any single failure mode collapsing**, even when the aggregate holds —
  "accuracy fell 4 points" tells an on-call engineer nothing; "null-handling
  collapsed while arithmetic held" points at the change

Recorded baseline on this hardware: **77.5% field accuracy, 100% schema
validity, 11.7% omission, 0.0% commission.** The omission-dominates asymmetry
Project 2 documented reproduces exactly.

### About the CI badge

Two jobs, because only one can run on hosted infrastructure and the difference
matters:

| Job | Runner | Catches | Does **not** catch |
|---|---|---|---|
| `replay` | GitHub-hosted | bugs in the scorer, thresholds, gate arithmetic | model or config regressions — the model is not re-run |
| `live` | self-hosted + GPU | swapped model tag, quantisation change, prompt regression | — |

A green check from `replay` alone does not mean the model is fine. The job names
say so rather than implying otherwise.

`tests/` covers the gate's own false-negative cases — a gate with a bug that
makes it always pass is worse than no gate, because it is also a green check.
One of those tests found a real defect: the scorer keyed results by task id, so
a duplicate row silently replaced rather than averaged.

---

## Running it

Requires Python 3.12 (Phoenix's trace-filter dataclass is invalid on 3.11), a
running Ollama, and one free GPU.

```bash
pip install -r requirements.txt
./scripts/stack.sh up        # phoenix :6006 · prometheus :9090 · grafana :3000 · api :8100
```

```bash
# a single extraction, graded against a gold label
curl -sX POST localhost:8100/v1/extract -H 'content-type: application/json' -d '{
  "text": "Today is 2026-03-09. Priya: I'\''ll get the migration script merged by March 11th.",
  "schema_name": "action_item", "task_id": "ai-01"}'

python scripts/loadgen.py --mode open --rate 1 --duration 60   # populate the dashboards
python scripts/incident.py                                     # reproduce the incident (~12 min)
python eval/gate.py --record                                   # (re)record the baseline
python eval/gate.py                                            # gate against it
```

### No Docker

The spec calls for self-hosted Langfuse via `docker-compose`. The host this was
built on grants no Docker access — no `docker` group, no passwordless sudo — so
[`scripts/stack.sh`](scripts/stack.sh) runs the stack as four unprivileged
user-space processes instead of four containers.

That constraint was worth keeping. Instrumentation is written against the
OpenTelemetry SDK and the Prometheus exposition format rather than a vendor
client, so Phoenix and Grafana are interchangeable parts and swapping backends
is an endpoint change. `ops/docker-compose.yml` ships the containerised
equivalent, labelled as unexercised, because it is.

### Load-bearing details

- **The VRAM device is detected, not assumed.**
  [`scripts/detect_gpu.py`](scripts/detect_gpu.py) evicts the model, snapshots
  every device, reloads, and takes the one that grew. On this host devices 0
  and 1 are saturated by another tenant and Ollama places models on 2 —
  sampling device 0 would have reported 46 GB of somebody else's memory as this
  service's footprint.
- **The load generator's arrival model defaults to open, not closed.** A closed
  loop is self-limiting: queue depth cannot grow, so P95 stays flat however slow
  the service gets. That is why most naive load scripts never reproduce a
  queueing incident.
- **`scripts/incident.py` waits for Ollama to drain between phases.** Cancelling
  a request does not cancel the generation behind it, so the GPU keeps working
  for minutes after the client gives up. Skipping the drain measures the
  previous phase — which is exactly how the first attempt produced a nonsensical
  19s "steady state".

---

## Limitations

- **Single model, single GPU, single host.** The queueing behaviour is specific
  to one-model-one-GPU; a multi-replica deployment has a different failure mode.
- **Quality metrics cover graded canary traffic only.** Organic traffic has no
  labels. This is stated rather than papered over, but it does mean online
  accuracy is a sample, not a census.
- **The hosted-cost counterfactual compares Ollama's token counts against
  another vendor's tokenizer and prices.** Close enough to size a decision, not
  close enough to reconcile an invoice. Prices were current on 2026-08-08 and
  will drift.
- **One alert is knowingly miscalibrated and left that way.**
  `GradedAccuracyRegression` fires in all three incident phases, including both
  healthy ones: its threshold was derived from an aggregate baseline but is
  evaluated per `(model, schema, mode)`, so schemas below the mean fire forever.
  It is documented as an open action item rather than quietly retuned, because
  the failure — an always-firing alert that trains the on-call to ignore the
  channel — is the more instructive artefact.
- **No dashboard screenshots.** Grafana's image renderer is a separate plugin
  that is not installed; `reports/incident_timeline.json` is the durable record,
  and it is the source of every number quoted here.
