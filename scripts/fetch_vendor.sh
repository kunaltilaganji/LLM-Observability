#!/usr/bin/env bash
# Download Prometheus and Grafana into ops/vendor/ (gitignored).
# These are the two components stack.sh runs from user space instead of Docker.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V="$ROOT/ops/vendor"; mkdir -p "$V"; cd "$V"

PROM_VERSION="${PROM_VERSION:-3.13.2}"
GRAFANA_VERSION="${GRAFANA_VERSION:-13.1.3}"

if [[ ! -x prometheus/prometheus ]]; then
  echo "fetching prometheus ${PROM_VERSION}..."
  curl -fsSL -o p.tgz "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz"
  tar xzf p.tgz && rm p.tgz && mv "prometheus-${PROM_VERSION}.linux-amd64" prometheus
fi

if [[ ! -x grafana/bin/grafana ]]; then
  echo "fetching grafana ${GRAFANA_VERSION}... (~150MB)"
  curl -fsSL -o g.tgz "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
  tar xzf g.tgz && rm g.tgz && mv "grafana-v${GRAFANA_VERSION}" grafana 2>/dev/null || mv "grafana-${GRAFANA_VERSION}" grafana
fi

echo "vendored:"; ./prometheus/prometheus --version 2>&1 | head -1; ./grafana/bin/grafana --version
