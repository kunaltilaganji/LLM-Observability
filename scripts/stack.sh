#!/usr/bin/env bash
# Start, stop, and check the observability stack.
#
# This is the docker-compose.yml of this repo. The host it was built on does not
# grant Docker access -- no docker group, no passwordless sudo -- so the stack
# runs as four unprivileged user-space processes instead of four containers.
#
# That constraint turned out to be worth keeping rather than working around:
# instrumentation is written against the OpenTelemetry SDK and the Prometheus
# exposition format, not against a vendor's client library, so Phoenix and
# Grafana are interchangeable parts. `ops/docker-compose.yml` ships the
# containerised equivalent for anyone who can run it; it is marked as
# unexercised because it is.
#
#   ./scripts/stack.sh up      start everything
#   ./scripts/stack.sh down    stop everything
#   ./scripts/stack.sh status  what is listening, and is it healthy
#   ./scripts/stack.sh logs    tail all four logs

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/ops/vendor"
RUN="$ROOT/.run"
LOGS="$ROOT/logs"
PY="${LLMOBS_PYTHON:-/data1/kunal.prjt/miniconda3/envs/llmobs312/bin/python}"

SERVICE_PORT="${LLMOBS_PORT:-8100}"
PROM_PORT="${LLMOBS_PROM_PORT:-9090}"
GRAFANA_PORT="${LLMOBS_GRAFANA_PORT:-3000}"
PHOENIX_PORT="${LLMOBS_PHOENIX_PORT:-6006}"

mkdir -p "$RUN" "$LOGS"

_pidfile() { echo "$RUN/$1.pid"; }

_running() {
  local pf; pf="$(_pidfile "$1")"
  [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null
}

_start() {          # _start <name> <logfile> <command...>
  local name="$1" log="$2"; shift 2
  if _running "$name"; then
    echo "  $name already running (pid $(cat "$(_pidfile "$name")"))"
    return 0
  fi
  "$@" >"$log" 2>&1 &
  echo $! > "$(_pidfile "$name")"
  echo "  $name started (pid $!) -> $log"
}

_wait_http() {      # _wait_http <url> <seconds> <label>
  local url="$1" deadline=$(( SECONDS + $2 )) label="$3"
  while (( SECONDS < deadline )); do
    if curl -sf -o /dev/null --max-time 2 "$url"; then
      echo "  $label ready"; return 0
    fi
    sleep 1
  done
  echo "  WARNING: $label did not answer at $url within $2s (see logs/)"
  return 1
}

up() {
  echo "Starting observability stack"

  # Phoenix first: it owns the OTLP collector endpoint, and the service export
  # pipeline is fire-and-forget, so a late collector silently drops the first
  # traces rather than erroring.
  # Phoenix takes its port and storage location from the environment, not
  # from flags -- `phoenix serve --port` is not a thing.
  _start phoenix "$LOGS/phoenix.log" \
    env PHOENIX_PORT="$PHOENIX_PORT" \
        PHOENIX_HOST=0.0.0.0 \
        PHOENIX_WORKING_DIR="$RUN/phoenix-data" \
        "$PY" -m phoenix.server.main serve
  _wait_http "http://localhost:$PHOENIX_PORT/" 90 "phoenix (traces, :$PHOENIX_PORT)"

  _start prometheus "$LOGS/prometheus.log" \
    "$VENDOR/prometheus/prometheus" \
      --config.file="$ROOT/ops/prometheus/prometheus.yml" \
      --storage.tsdb.path="$RUN/prometheus-data" \
      --web.listen-address="0.0.0.0:$PROM_PORT" \
      --web.enable-lifecycle
  _wait_http "http://localhost:$PROM_PORT/-/ready" 30 "prometheus (:$PROM_PORT)"

  if [[ -x "$VENDOR/grafana/bin/grafana" ]]; then
    _start grafana "$LOGS/grafana.log" \
      env GF_PATHS_DATA="$RUN/grafana-data" \
          GF_PATHS_LOGS="$LOGS/grafana" \
          GF_PATHS_PLUGINS="$RUN/grafana-plugins" \
          GF_PATHS_PROVISIONING="$ROOT/ops/grafana/provisioning" \
          LLMOBS_DASHBOARD_DIR="$ROOT/ops/grafana/dashboards" \
          GF_SERVER_HTTP_PORT="$GRAFANA_PORT" \
          GF_AUTH_ANONYMOUS_ENABLED=true \
          GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
          GF_SECURITY_ADMIN_PASSWORD=admin \
          "$VENDOR/grafana/bin/grafana" server --homepath "$VENDOR/grafana"
    _wait_http "http://localhost:$GRAFANA_PORT/api/health" 60 "grafana (:$GRAFANA_PORT)"
  else
    echo "  grafana not vendored; skipping (run scripts/fetch_vendor.sh)"
  fi

  # Which GPU to sample for the VRAM metric. Measured rather than assumed:
  # device 0 is the wrong answer on any host where another tenant holds it.
  if [[ -z "${LLMOBS_GPU_INDEX:-}" ]]; then
    LLMOBS_GPU_INDEX="$("$PY" "$ROOT/scripts/detect_gpu.py" \
      --model "${LLMOBS_MODEL:-gemma3:4b}" 2>>"$LOGS/detect_gpu.log")"
    echo "  VRAM sampling device: GPU ${LLMOBS_GPU_INDEX} (detected; see logs/detect_gpu.log)"
  fi

  _start service "$LOGS/service.log" \
    env PYTHONPATH="$ROOT" \
        OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:$PHOENIX_PORT/v1/traces" \
        LLMOBS_GPU_INDEX="$LLMOBS_GPU_INDEX" \
        "$PY" -m uvicorn llmobs.app:app --host 0.0.0.0 --port "$SERVICE_PORT"
  _wait_http "http://localhost:$SERVICE_PORT/healthz" 60 "extraction service (:$SERVICE_PORT)"

  echo
  status
}

down() {
  echo "Stopping observability stack"
  for name in service grafana prometheus phoenix; do
    local pf; pf="$(_pidfile "$name")"
    if _running "$name"; then
      local pid; pid="$(cat "$pf")"
      # SIGTERM first so Prometheus checkpoints its WAL and the service flushes
      # pending spans; a SIGKILL here loses the tail of every trace.
      kill "$pid" 2>/dev/null
      for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
      kill -0 "$pid" 2>/dev/null && { echo "  $name did not exit; SIGKILL"; kill -9 "$pid" 2>/dev/null; }
      echo "  $name stopped"
    else
      echo "  $name not running"
    fi
    rm -f "$pf"
  done
}

status() {
  printf '%-12s %-9s %-34s %s\n' COMPONENT STATE URL NOTE
  _row() { printf '%-12s %-9s %-34s %s\n' "$1" "$2" "$3" "$4"; }
  for entry in \
    "phoenix:$PHOENIX_PORT:/:traces (OpenInference)" \
    "prometheus:$PROM_PORT:/-/ready:metrics + alert rules" \
    "grafana:$GRAFANA_PORT:/api/health:dashboards" \
    "service:$SERVICE_PORT:/healthz:extraction API"; do
    IFS=: read -r name port path note <<<"$entry"
    local url="http://localhost:$port"
    if _running "$name" && curl -sf -o /dev/null --max-time 2 "$url$path"; then
      _row "$name" "up" "$url" "$note"
    elif _running "$name"; then
      _row "$name" "starting" "$url" "$note"
    else
      _row "$name" "down" "$url" "$note"
    fi
  done
}

logs() { tail -n 40 -F "$LOGS"/*.log; }

case "${1:-up}" in
  up) up ;;
  down) down ;;
  restart) down; sleep 2; up ;;
  status) status ;;
  logs) logs ;;
  *) echo "usage: $0 {up|down|restart|status|logs}"; exit 2 ;;
esac
