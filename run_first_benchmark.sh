#!/usr/bin/env bash
set -Eeuo pipefail

# Conservative defaults for an Apple Silicon desktop or a shared machine.
DATA_ROOT=${1:?Usage: bash run_first_benchmark.sh /absolute/path/to/nyuv2 [run_dir]}
RUN_DIR=${2:-./runs_m5}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
WORKERS=${WORKERS:-0}
DEVICE=${DEVICE:-auto}
CPU_THREADS=${CPU_THREADS:-6}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.25}
IMAGE_NORMALIZATION=${IMAGE_NORMALIZATION:-imagenet}
VAL_EVERY=${VAL_EVERY:-2}
AMP=${AMP:-1}
NICE_LEVEL=${NICE_LEVEL:-0}
LR=${LR:-2e-4}

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Prevent accidental duplicate benchmark runs with a portable mkdir lock.
# macOS does not ship flock(1).
TMP_BASE=${TMPDIR:-/tmp}
LOCK_DIR="${TMP_BASE%/}/nyuv2_merge_benchmark_${USER:-$(id -u)}.lock"

cleanup_lock() {
  if [ -f "$LOCK_DIR/pid" ] && [ "$(cat "$LOCK_DIR/pid" 2>/dev/null)" = "$$" ]; then
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  OWNER_PID=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
  if [ -n "$OWNER_PID" ] && ! kill -0 "$OWNER_PID" 2>/dev/null; then
    echo "Removing stale benchmark lock from PID $OWNER_PID."
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another NYUv2 benchmark is already running for user ${USER:-$(id -u)}; exiting." >&2
    exit 2
  fi
fi
printf '%s\n' "$$" >"$LOCK_DIR/pid"
trap cleanup_lock EXIT

# Optional priority reduction for shared machines. Default mode runs at
# normal priority so local DataLoader workers are not starved.
run_benchmark() {
  if [ "$NICE_LEVEL" -gt 0 ]; then
    nice -n "$NICE_LEVEL" "$@"
  else
    "$@"
  fi
}

if [ "$AMP" = "0" ]; then
  AMP_ARGS=(--no-amp)
else
  AMP_ARGS=(--amp)
fi

COMMON=(--data_root "$DATA_ROOT" --out_dir "$RUN_DIR" --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" --workers "$WORKERS" --device "$DEVICE" \
        --cpu_threads "$CPU_THREADS" --gpu_memory_fraction "$GPU_MEMORY_FRACTION" \
        --image_normalization "$IMAGE_NORMALIZATION" --val_every "$VAL_EVERY" \
        --lr "$LR" "${AMP_ARGS[@]}")

echo "[benchmark] device=$DEVICE normalization=$IMAGE_NORMALIZATION batch=$BATCH_SIZE workers=$WORKERS cpu_threads=$CPU_THREADS amp=$AMP val_every=$VAL_EVERY"
echo "[benchmark] tasks are strictly sequential; output=$RUN_DIR"

run_benchmark python train.py --mode stl --task segmentation "${COMMON[@]}"
run_benchmark python train.py --mode stl --task depth        "${COMMON[@]}"
run_benchmark python train.py --mode stl --task normal       "${COMMON[@]}"
run_benchmark python train.py --mode mtl                     "${COMMON[@]}"

run_benchmark python merge_average.py --run_dir "$RUN_DIR" --out "$RUN_DIR/merged_average.pth"
run_benchmark python evaluate.py --data_root "$DATA_ROOT" --run_dir "$RUN_DIR" \
  --merged "$RUN_DIR/merged_average.pth" --csv "$RUN_DIR/benchmark_results.csv" \
  --batch_size "$EVAL_BATCH_SIZE" --workers "$WORKERS" --device "$DEVICE" \
  --cpu_threads "$CPU_THREADS" --gpu_memory_fraction "$GPU_MEMORY_FRACTION" \
  --image_normalization "$IMAGE_NORMALIZATION" "${AMP_ARGS[@]}"

echo "Benchmark completed: $RUN_DIR/benchmark_results.csv"
