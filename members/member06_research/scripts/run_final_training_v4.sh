#!/usr/bin/env bash
# User-started runner for Baseline 2 final_training_v4 6 core experiments:
#   V4-R1_RAW_CE (seeds 42, 43, 44) & V4-R2_RAW_WEIGHTED_CE (seeds 42, 43, 44), 20 epochs each.
# Never modifies final_training_v4, never touches Test, never overwrites prior runs.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

BASE="members/member06_research"
V4_DIR="members/member03_data/final_training_v4"
PY="$ROOT/$BASE/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing virtual environment interpreter: $PY" >&2; exit 2; }

export TORCH_HOME="$ROOT/$BASE/.venv/torch"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/member06-v4-mpl"

ALL_KEYS=(R1_S42 R1_S43 R1_S44 R2_S42 R2_S43 R2_S44)

usage() {
    cat >&2 <<'EOF'
Usage:
  bash members/member06_research/scripts/run_final_training_v4.sh --check-only
  bash members/member06_research/scripts/run_final_training_v4.sh [--suite-dir DIR]
  bash members/member06_research/scripts/run_final_training_v4.sh --only R1_S42|R1_S43|R1_S44|R2_S42|R2_S43|R2_S44 [--suite-dir DIR]
EOF
    exit 2
}

spec_for() {
    case "$1" in
        R1_S42) printf '%s/configs/final_training_v4/v4_r1_raw_ce_seed42.json' "$BASE" ;;
        R1_S43) printf '%s/configs/final_training_v4/v4_r1_raw_ce_seed43.json' "$BASE" ;;
        R1_S44) printf '%s/configs/final_training_v4/v4_r1_raw_ce_seed44.json' "$BASE" ;;
        R2_S42) printf '%s/configs/final_training_v4/v4_r2_raw_weighted_ce_seed42.json' "$BASE" ;;
        R2_S43) printf '%s/configs/final_training_v4/v4_r2_raw_weighted_ce_seed43.json' "$BASE" ;;
        R2_S44) printf '%s/configs/final_training_v4/v4_r2_raw_weighted_ce_seed44.json' "$BASE" ;;
        *) return 2 ;;
    esac
}

name_for() {
    case "$1" in
        R1_S42) printf 'V4-R1_RAW_CE_SEED42' ;;
        R1_S43) printf 'V4-R1_RAW_CE_SEED43' ;;
        R1_S44) printf 'V4-R1_RAW_CE_SEED44' ;;
        R2_S42) printf 'V4-R2_RAW_WEIGHTED_CE_SEED42' ;;
        R2_S43) printf 'V4-R2_RAW_WEIGHTED_CE_SEED43' ;;
        R2_S44) printf 'V4-R2_RAW_WEIGHTED_CE_SEED44' ;;
        *) return 2 ;;
    esac
}

MODE="all"
ONLY_KEY=""
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
            ONLY_KEY="$2"
            name_for "$ONLY_KEY" >/dev/null || usage
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

if [[ "$MODE" == "check" ]]; then
    echo "=== Upstream final_training_v4 verify_dataset.py (Train+Val 283 triples) ==="
    "$PY" "$V4_DIR/verify_dataset.py"
    for k in "${ALL_KEYS[@]}"; do
        echo "=== Preflight $(name_for "$k") ==="
        "$PY" -u -m members.member06_research.v4_train --spec "$(spec_for "$k")" --check-only
    done
    exit 0
fi

echo "=== Verifying final_training_v4 Train/Val integrity before run ==="
"$PY" "$V4_DIR/verify_dataset.py"

mkdir -p "$BASE/runs"
if [[ -z "$SUITE_DIR" ]]; then
    SUITE_DIR="$(mktemp -d "$BASE/runs/final_training_v4_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
else
    mkdir -p "$SUITE_DIR"
fi
SUITE_ABS="$(cd "$SUITE_DIR" && pwd)"
printf 'final_training_v4 Suite directory: %s\n' "$SUITE_ABS"

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

    "$PY" -u -m members.member06_research.v4_train --spec "$spec_path" --check-only --check-output "$exp_dir/preflight.json" > "$exp_dir/preflight.log" 2>&1

    set +e
    "$PY" -u -m members.member06_research.v4_train --spec "$spec_path" --output "$exp_dir/model" 2>&1 | tee "$exp_dir/console.log"
    local codes=("${PIPESTATUS[@]}")
    set -e

    printf '%s\n' "${codes[0]}" > "$exp_dir/train_exit_code.txt"
    [[ ${codes[0]} -eq 0 ]] || exit "${codes[0]}"
    [[ ${codes[1]} -eq 0 ]] || { echo "Log capture failed for $exp_name" >&2; exit "${codes[1]}"; }
    printf '=== Completed %s (%s/model/) ===\n' "$exp_name" "$exp_dir"
}

if [[ "$MODE" == "single" ]]; then
    run_one "$ONLY_KEY"
else
    for k in "${ALL_KEYS[@]}"; do
        run_one "$k"
    done
fi

# Auto-generate unified comparison report once all 6 experiments have completed
all_done=1
for k in "${ALL_KEYS[@]}"; do
    if [[ ! -f "$SUITE_ABS/$(name_for "$k")/model/best_within_20.pt" ]]; then
        all_done=0
        break
    fi
done

if [[ "$all_done" -eq 1 && ! -e "$SUITE_ABS/comparison" ]]; then
    printf '=== Generating Unified V4 Comparison Report -> %s/comparison ===\n' "$SUITE_ABS"
    "$PY" -u -m members.member06_research.v4_compare \
        --suite-dir "$SUITE_ABS" \
        --output "$SUITE_ABS/comparison" 2>&1 | tee "$SUITE_ABS/comparison.log"
    printf 'V4 comparison report ready: %s/comparison/COMPARISON_V4_CN.md\n' "$SUITE_ABS"
fi
