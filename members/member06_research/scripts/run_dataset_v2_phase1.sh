#!/usr/bin/env bash
# Dataset v2 Phase-1 user-started training & comparison runner (R1 / R2 / R3).
# Never modifies Dataset v2, never touches Test, never retries or alters parameters on failure.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

BASE="members/member06_research"
PY="$ROOT/$BASE/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing virtual environment interpreter: $PY" >&2; exit 2; }

export TORCH_HOME="$ROOT/$BASE/.venv/torch"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/member06-v2-mpl"

SPEC_R1="$BASE/configs/dataset_v2/r1_raw_ce.json"
SPEC_R2="$BASE/configs/dataset_v2/r2_parent_equal_aug_ce.json"
SPEC_R3="$BASE/configs/dataset_v2/r3_full_aug_balanced_ce.json"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash members/member06_research/scripts/run_dataset_v2_phase1.sh --check-only
  bash members/member06_research/scripts/run_dataset_v2_phase1.sh [--suite-dir DIR]
  bash members/member06_research/scripts/run_dataset_v2_phase1.sh --only R1|R2|R3 [--suite-dir DIR]
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
            [[ "$ONLY_EXP" == "R1" || "$ONLY_EXP" == "R2" || "$ONLY_EXP" == "R3" ]] || usage
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
        R1) printf '%s' "$SPEC_R1" ;;
        R2) printf '%s' "$SPEC_R2" ;;
        R3) printf '%s' "$SPEC_R3" ;;
        *) return 2 ;;
    esac
}

name_for() {
    case "$1" in
        R1) printf 'R1_RAW_CE' ;;
        R2) printf 'R2_PARENT_EQUAL_AUG_CE' ;;
        R3) printf 'R3_FULL_AUG_BALANCED_CE' ;;
        *) return 2 ;;
    esac
}

if [[ "$MODE" == "check" ]]; then
    for k in R1 R2 R3; do
        echo "=== Preflight $(name_for "$k") ==="
        "$PY" -u -m members.member06_research.v2_train --spec "$(spec_for "$k")" --check-only
    done
    exit 0
fi

mkdir -p "$BASE/runs"
if [[ -z "$SUITE_DIR" ]]; then
    SUITE_DIR="$(mktemp -d "$BASE/runs/dataset_v2_phase1_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
else
    mkdir -p "$SUITE_DIR"
fi
SUITE_ABS="$(cd "$SUITE_DIR" && pwd)"
printf 'Suite directory: %s\n' "$SUITE_ABS"

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
    printf '=== Completed %s (best.pt: %s/model/best.pt) ===\n' "$exp_name" "$exp_dir"
}

if [[ "$MODE" == "single" ]]; then
    run_one "$ONLY_EXP"
else
    run_one R1
    run_one R2
    run_one R3
fi

# Automatically generate 3-way comparison report whenever all 3 models exist in SUITE_ABS and comparison/ does not exist yet
if [[ -f "$SUITE_ABS/R1_RAW_CE/model/best.pt" && -f "$SUITE_ABS/R2_PARENT_EQUAL_AUG_CE/model/best.pt" && -f "$SUITE_ABS/R3_FULL_AUG_BALANCED_CE/model/best.pt" && ! -e "$SUITE_ABS/comparison" ]]; then
    printf '=== Generating Phase-1 Comparison Report -> %s/comparison ===\n' "$SUITE_ABS"
    "$PY" -u -m members.member06_research.v2_compare \
        --r1 "$SUITE_ABS/R1_RAW_CE/model" \
        --r2 "$SUITE_ABS/R2_PARENT_EQUAL_AUG_CE/model" \
        --r3 "$SUITE_ABS/R3_FULL_AUG_BALANCED_CE/model" \
        --output "$SUITE_ABS/comparison" 2>&1 | tee "$SUITE_ABS/comparison.log"
    printf 'Comparison report ready: %s/comparison/COMPARISON_PHASE1_CN.md\n' "$SUITE_ABS"
fi
