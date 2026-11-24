#!/usr/bin/env bash
# Stage 2 official single-pass Test evaluation launcher for Member 6 Baseline 2.
# Usage:
#   Preflight only (no inference):
#     bash members/member06_research/scripts/run_stage2_final_test.sh --check-only
#   Formal Test evaluation (to be launched personally by Member 6):
#     bash members/member06_research/scripts/run_stage2_final_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMBER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MEMBER_DIR}/../.." && pwd)"
VENV_PY="${MEMBER_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing virtual environment interpreter: ${VENV_PY}" >&2
  exit 2
fi

export TORCH_HOME="${MEMBER_DIR}/.venv/torch"
export MPLCONFIGDIR="${MEMBER_DIR}/.venv/matplotlib"

cd "${REPO_ROOT}"

if [[ "${1:-}" == "--check-only" ]]; then
  "${VENV_PY}" -u -m members.member06_research.stage2_test_eval test --check-only
  exit 0
fi

OUT_DIR="${1:-${MEMBER_DIR}/stage2_release/test_evaluation_v4_official}"

echo "=== [1/2] Stage 2 Freeze & Test Manifest Integrity Preflight ==="
"${VENV_PY}" -u -m members.member06_research.stage2_test_eval test --check-only

echo "=== [2/2] Stage 2 Single-Pass Blind Inference & Authoritative Test Evaluation ==="
echo "Output directory: ${OUT_DIR}"
"${VENV_PY}" -u -m members.member06_research.stage2_test_eval test --output-dir "${OUT_DIR}"

echo "=== Completed! Official Test outputs saved to: ${OUT_DIR} ==="
echo "Report: ${OUT_DIR}/STAGE2_FINAL_TEST_REPORT_CN.md"
