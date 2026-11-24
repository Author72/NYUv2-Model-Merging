#!/usr/bin/env bash
set -Eeuo pipefail

DATA_ROOT=${1:?Usage: bash run_nvidia_three_seed.sh /absolute/path/to/nyuv2 [output_root]}
OUTPUT_ROOT=${2:-./runs_nvidia_three_seed_100ep}
PYTHON_BIN=${PYTHON_BIN:-python}
SEEDS=${SEEDS:-8394,4197,9424}
EPOCHS=${EPOCHS:-100}
DISTILL_EPOCHS=${DISTILL_EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
WORKERS=${WORKERS:-4}
CPU_THREADS=${CPU_THREADS:-8}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.90}
GPU_MEMORY_LIMIT_GB=${GPU_MEMORY_LIMIT_GB:-9.5}
REUSE_EXISTING=${REUSE_EXISTING:-0}
LR=${LR:-2e-4}
INIT_ENCODER=${INIT_ENCODER:-}

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Check the NVIDIA driver and install a CUDA "
        "PyTorch wheel before running this script."
    )
properties = torch.cuda.get_device_properties(0)
print(
    f"[preflight] CUDA available: {properties.name}; "
    f"capability={properties.major}.{properties.minor}; "
    f"memory={properties.total_memory / 1024**3:.1f} GB"
)
PY

if [ -n "$GPU_MEMORY_LIMIT_GB" ]; then
  GPU_MEMORY_FRACTION=$(
    "$PYTHON_BIN" -c \
      "import sys, torch; limit_gib=float(sys.argv[1]); total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3; print(min(1.0, limit_gib/total_gib))" \
      "$GPU_MEMORY_LIMIT_GB"
  )
fi

COMMON=(
  --data_root "$DATA_ROOT"
  --output_root "$OUTPUT_ROOT"
  --seeds "$SEEDS"
  --epochs "$EPOCHS"
  --distill_epochs "$DISTILL_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --lr "$LR"
  --workers "$WORKERS"
  --cpu_threads "$CPU_THREADS"
  --gpu_memory_fraction "$GPU_MEMORY_FRACTION"
  --device cuda
  --amp
)

if [ -n "$INIT_ENCODER" ]; then
  COMMON+=(--init_encoder "$INIT_ENCODER")
fi
if [ "$REUSE_EXISTING" = "1" ]; then
  COMMON+=(--reuse_existing_checkpoints)
fi

echo "[nvidia] seeds=$SEEDS batch=$BATCH_SIZE lr=$LR workers=$WORKERS gpu_memory_limit=${GPU_MEMORY_LIMIT_GB:-unset}GB gpu_memory_fraction=$GPU_MEMORY_FRACTION"
echo "[nvidia] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}; output=$OUTPUT_ROOT"
"$PYTHON_BIN" run_three_seed_benchmark.py "${COMMON[@]}"
