#!/usr/bin/env bash
# Dataset v2 Phase-2 user-started training & comparison runner (R4_RAW_CE_SEED42 / SEED43 / SEED44).
# Reuses existing verified 224 frozen feature cache; saves 10ep and 20ep budget checkpoints; never touches Test.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

BASE="members/member06_research"
PY="$ROOT/$BASE/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing virtual environment interpreter: $PY" >&2; exit 2; }

export TORCH_HOME="$ROOT/$BASE/.venv/torch"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/member06-v2-phase2-mpl"

SPEC_42="$BASE/configs/dataset_v2/r4_raw_ce_seed42.json"
SPEC_43="$BASE/configs/dataset_v2/r4_raw_ce_seed43.json"
SPEC_44="$BASE/configs/dataset_v2/r4_raw_ce_seed44.json"
PHASE1_SUITE="$BASE/runs/dataset_v2_phase1_20261124_124710_5rtNiM"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash members/member06_research/scripts/run_dataset_v2_phase2.sh --check-only
  bash members/member06_research/scripts/run_dataset_v2_phase2.sh [--suite-dir DIR]
  bash members/member06_research/scripts/run_dataset_v2_phase2.sh --only SEED42|SEED43|SEED44 [--suite-dir DIR]
EOF
    exit 2
}

MODE="all"
ONLY_EXP=""
SUITE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only)
            [[ $# -eq 1 && "$MODE" == "all" && -z "$SUITE_DIR" ]] || usage
            MODE="check"
            shift
            ;;
        --only)
            [[ $# -ge 2 ]] || usage
            MODE="single"
            ONLY_EXP="$2"
            [[ "$ONLY_EXP" == "SEED42" || "$ONLY_EXP" == "SEED43" || "$ONLY_EXP" == "SEED44" ]] || usage
            shift 2
            ;;
        --suite-dir)
            [[ $# -ge 2 ]] || usage
            SUITE_DIR="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

spec_for() {
    case "$1" in
        SEED42) printf '%s' "$SPEC_42" ;;
        SEED43) printf '%s' "$SPEC_43" ;;
        SEED44) printf '%s' "$SPEC_44" ;;
        *) return 2 ;;
    esac
}

name_for() {
    case "$1" in
        SEED42) printf 'R4_RAW_CE_SEED42' ;;
        SEED43) printf 'R4_RAW_CE_SEED43' ;;
        SEED44) printf 'R4_RAW_CE_SEED44' ;;
        *) return 2 ;;
    esac
}

if [[ "$MODE" == "check" ]]; then
    for k in SEED42 SEED43 SEED44; do
        echo "=== Preflight $(name_for "$k") ==="
        "$PY" -u -m members.member06_research.v2_train --spec "$(spec_for "$k")" --check-only
    done
    exit 0
fi

mkdir -p "$BASE/runs"
if [[ -z "$SUITE_DIR" ]]; then
    SUITE_DIR="$(mktemp -d "$BASE/runs/dataset_v2_phase2_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
else
    mkdir -p "$SUITE_DIR"
fi
SUITE_ABS="$(cd "$SUITE_DIR" && pwd)"
printf 'Phase-2 Suite directory: %s\n' "$SUITE_ABS"

run_one() {
    local key="$1"
    local exp_name spec_path exp_dir
    exp_name="$(name_for "$key")"
    spec_path="$(spec_for "$key")"
    exp_dir="$SUITE_ABS/$exp_name"

    if [[ -e "$exp_dir" ]]; then
        echo "Refusing to overwrite existing experiment directory: $exp_dir" >&2
        exit 2
    fi
    mkdir -p "$exp_dir"
    printf '=== Starting %s -> %s ===\n' "$exp_name" "$exp_dir"

    "$PY" -u -m members.member06_research.v2_train --spec "$spec_path" --check-only --check-output "$exp_dir/preflight.json" > "$exp_dir/preflight.log" 2>&1

    set +e
    "$PY" -u -m members.member06_research.v2_train --spec "$spec_path" --output "$exp_dir/model" --allow-dev-only 2>&1 | tee "$exp_dir/console.log"
    local codes=("${PIPESTATUS[@]}")
    set -e

    printf '%s\n' "${codes[0]}" > "$exp_dir/train_exit_code.txt"
    [[ ${codes[0]} -eq 0 ]] || exit "${codes[0]}"
    [[ ${codes[1]} -eq 0 ]] || { echo "Log capture failed for $exp_name" >&2; exit "${codes[1]}"; }
    printf '=== Completed %s (best_within_10.pt & best_within_20.pt ready in %s/model/) ===\n' "$exp_name" "$exp_dir"
}

if [[ "$MODE" == "single" ]]; then
    run_one "$ONLY_EXP"
else
    run_one SEED42
    run_one SEED43
    run_one SEED44
fi

# Automatically generate Phase-2 10ep-vs-20ep & multi-seed comparison report once all 3 runs exist
if [[ -f "$SUITE_ABS/R4_RAW_CE_SEED42/model/best_within_20.pt" && \
      -f "$SUITE_ABS/R4_RAW_CE_SEED43/model/best_within_20.pt" && \
      -f "$SUITE_ABS/R4_RAW_CE_SEED44/model/best_within_20.pt" && \
      ! -e "$SUITE_ABS/comparison" ]]; then
    printf '=== Generating Phase-2 Comparison Report -> %s/comparison ===\n' "$SUITE_ABS"
    "$PY" -u -m members.member06_research.v2_phase2_compare \
        --seed42 "$SUITE_ABS/R4_RAW_CE_SEED42/model" \
        --seed43 "$SUITE_ABS/R4_RAW_CE_SEED43/model" \
        --seed44 "$SUITE_ABS/R4_RAW_CE_SEED44/model" \
        --phase1-suite "$PHASE1_SUITE" \
        --output "$SUITE_ABS/comparison" 2>&1 | tee "$SUITE_ABS/comparison.log"
    printf 'Phase-2 comparison report ready: %s/comparison/COMPARISON_PHASE2_CN.md\n' "$SUITE_ABS"
fi
