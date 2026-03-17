#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_embedded_subspace_experiments.sh \
    --experiment embedded_subspace \
    --base sphere \
    --ambient 50 \
    --intrinsic 5

Notes:
  - Runs 5 seeds (0-4) for each optimizer: tpe, cma_es, diffusion.
  - Logs go to outputs/logs/.
EOF
}

experiment="embedded_subspace"
base=""
ambient=""
intrinsic=""
log_root="outputs/logs"
optimizers=(tpe cma_es diffusion)
seeds=(0 1 2 3 4)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      experiment="$2"
      shift 2
      ;;
    --base)
      base="$2"
      shift 2
      ;;
    --ambient)
      ambient="$2"
      shift 2
      ;;
    --intrinsic)
      intrinsic="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${base}" || -z "${ambient}" || -z "${intrinsic}" ]]; then
  echo "Missing required arguments."
  usage
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${log_root}/${experiment}_${base}_D${ambient}_d${intrinsic}_${timestamp}"
mkdir -p "${run_dir}"

for seed in "${seeds[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    log_file="${run_dir}/${optimizer}_seed${seed}.log"
    python main.py \
      --benchmark "${experiment}" \
      --base-function "${base}" \
      --ambient-dim "${ambient}" \
      --intrinsic-dim "${intrinsic}" \
      --method "${optimizer}" \
      --seed "${seed}" \
      --output-dir outputs \
      > "${log_file}" 2>&1
  done
done
