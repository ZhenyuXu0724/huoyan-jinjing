#!/usr/bin/env bash
# User-start only. No retry, no parameter changes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
BASE=members/member06_research
PY="$ROOT/$BASE/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing virtual environment: $PY" >&2; exit 2; }
export TORCH_HOME="$ROOT/$BASE/.venv/torch"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/member06-weighted-mpl"
CONFIG="$BASE/configs/baseline2_weighted_ce_dev_v2.json"
if [[ "${1:-}" == "--check-only" && $# == 1 ]]; then
    exec "$PY" -u -m members.member06_research.weighted_dev --config "$CONFIG"
fi
[[ $# == 0 ]] || { echo 'Usage: bash run_weighted_dev.sh [--check-only]' >&2; exit 2; }
"$PY" -u -m members.member06_research.weighted_dev --config "$CONFIG"
mkdir -p "$BASE/runs"
RUN="$(mktemp -d "$BASE/runs/baseline2_weighted_ce_dev_v2_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
printf 'Experiment: %s\n' "$ROOT/$RUN"
"$PY" -u -m members.member06_research.weighted_dev --config "$CONFIG" --output "$RUN/preflight.json" > "$RUN/preflight.log" 2>&1
set +e
"$PY" -u -m members.member06_research.train --config "$CONFIG" --output "$RUN/model" --allow-dev-only 2>&1 | tee "$RUN/console.log"
CODES=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${CODES[0]}" > "$RUN/train_exit_code.txt"
printf 'Experiment: %s\nCheckpoint: %s\nLog: %s\n' "$ROOT/$RUN" "$ROOT/$RUN/model/best.pt" "$ROOT/$RUN/console.log"
[[ ${CODES[0]} == 0 ]] || exit "${CODES[0]}"
[[ ${CODES[1]} == 0 ]] || { echo 'Log capture failed' >&2; exit "${CODES[1]}"; }
"$PY" -u -m members.member06_research.compare_dev_runs --new-run "$RUN/model" --output "$RUN/comparison" 2>&1 | tee "$RUN/comparison.log"
