#!/usr/bin/env bash
# User-started runner for the final Baseline 2 optimization experiment:
#   V4-R4_RAW_WEIGHTED_CE_384 (seeds 42, 43, 44; 20 epochs; image_size=384)
# Pairs automatically against completed 224 V4-R2 runs and writes COMPARISON_V4_FINAL_RESOLUTION_CN.md.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

BASE="members/member06_research"
V4_DIR="members/member03_data/final_training_v4"
SUITE_224_DEFAULT="$BASE/runs/final_training_v4_20261124_192110_PJSHyF"
PY="$ROOT/$BASE/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing virtual environment interpreter: $PY" >&2; exit 2; }

export TORCH_HOME="$ROOT/$BASE/.venv/torch"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/member06-v4-384-mpl"

ALL_KEYS=(S42 S43 S44)

usage() {
    cat >&2 <<'EOF'
Usage:
  bash members/member06_research/scripts/run_final_training_v4_384.sh --check-only
  bash members/member06_research/scripts/run_final_training_v4_384.sh [--suite-dir DIR] [--suite-224 DIR]
  bash members/member06_research/scripts/run_final_training_v4_384.sh --only S42|S43|S44 [--suite-dir DIR] [--suite-224 DIR]
EOF
    exit 2
}

spec_for() {
    case "$1" in
        S42) printf '%s/configs/final_training_v4/v4_r4_raw_weighted_ce_384_seed42.json' "$BASE" ;;
        S43) printf '%s/configs/final_training_v4/v4_r4_raw_weighted_ce_384_seed43.json' "$BASE" ;;
        S44) printf '%s/configs/final_training_v4/v4_r4_raw_weighted_ce_384_seed44.json' "$BASE" ;;
        *) return 2 ;;
    esac
}

name_for() {
    case "$1" in
        S42) printf 'V4-R4_RAW_WEIGHTED_CE_384_SEED42' ;;
        S43) printf 'V4-R4_RAW_WEIGHTED_CE_384_SEED43' ;;
        S44) printf 'V4-R4_RAW_WEIGHTED_CE_384_SEED44' ;;
        *) return 2 ;;
    esac
}

MODE="all"
ONLY_KEY=""
SUITE_DIR=""
SUITE_224="$SUITE_224_DEFAULT"

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
        --suite-224)
            [[ $# -ge 2 ]] || usage
            SUITE_224="$2"
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
    [[ -d "$SUITE_224" ]] || { echo "Missing 224 reference suite: $SUITE_224" >&2; exit 2; }
    for k in "${ALL_KEYS[@]}"; do
        echo "=== Preflight $(name_for "$k") ==="
        "$PY" -u -m members.member06_research.v4_train --spec "$(spec_for "$k")" --check-only
    done
    exit 0
fi

[[ -d "$SUITE_224" ]] || { echo "Missing 224 reference suite: $SUITE_224" >&2; exit 2; }
echo "=== Verifying final_training_v4 Train/Val integrity before 384 run ==="
"$PY" "$V4_DIR/verify_dataset.py"

mkdir -p "$BASE/runs"
if [[ -z "$SUITE_DIR" ]]; then
    SUITE_DIR="$(mktemp -d "$BASE/runs/final_training_v4_384_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
else
    mkdir -p "$SUITE_DIR"
fi
SUITE_ABS="$(cd "$SUITE_DIR" && pwd)"
printf 'final_training_v4 384 Suite directory: %s\n' "$SUITE_ABS"

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

# Auto-generate final 224-vs-384 comparison report once all 3 seeds have completed
all_done=1
for k in "${ALL_KEYS[@]}"; do
    if [[ ! -f "$SUITE_ABS/$(name_for "$k")/model/best_within_20.pt" ]]; then
        all_done=0
        break
    fi
done

if [[ "$all_done" -eq 1 && ! -e "$SUITE_ABS/comparison" ]]; then
    printf '=== Generating Final 224-vs-384 Comparison Report -> %s/comparison ===\n' "$SUITE_ABS"
    "$PY" -u -m members.member06_research.v4_resolution_compare \
        --suite-224 "$SUITE_224" \
        --suite-384 "$SUITE_ABS" \
        --output "$SUITE_ABS/comparison" 2>&1 | tee "$SUITE_ABS/comparison.log"
    printf 'Final resolution comparison report ready: %s/comparison/COMPARISON_V4_FINAL_RESOLUTION_CN.md\n' "$SUITE_ABS"
fi
