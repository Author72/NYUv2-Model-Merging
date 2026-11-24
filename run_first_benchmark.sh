#!/usr/bin/env bash
set -Eeuo pipefail

# NVIDIA/Linux defaults. CUDA_VISIBLE_DEVICES can be set by the caller.
DATA_ROOT=${1:?Usage: bash run_first_benchmark.sh /absolute/path/to/nyuv2 [run_dir]}
RUN_DIR=${2:-./runs_nvidia}
PYTHON_BIN=${PYTHON_BIN:-python}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
WORKERS=${WORKERS:-4}
DEVICE=${DEVICE:-cuda}
CPU_THREADS=${CPU_THREADS:-8}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.90}
IMAGE_NORMALIZATION=${IMAGE_NORMALIZATION:-imagenet}
VAL_EVERY=${VAL_EVERY:-10}
SEED=${SEED:-0}
AMP=${AMP:-1}
NICE_LEVEL=${NICE_LEVEL:-0}
LR=${LR:-4e-4}
CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.10}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-}
DISTILL_EPOCHS=${DISTILL_EPOCHS:-$EPOCHS}
DISTILL_BATCH_SIZE=${DISTILL_BATCH_SIZE:-8}
DISTILL_LR=${DISTILL_LR:-1e-3}
DISTILL_VAL_EVERY=${DISTILL_VAL_EVERY:-10}
BN_CALIBRATION_PASSES=${BN_CALIBRATION_PASSES:-1}
RUN_HEAD_DISTILLATION=${RUN_HEAD_DISTILLATION:-1}
OUTPUT_DISTILL_EPOCHS=${OUTPUT_DISTILL_EPOCHS:-$EPOCHS}
OUTPUT_DISTILL_LR=${OUTPUT_DISTILL_LR:-1e-3}
INTERMEDIATE_DISTILL_EPOCHS=${INTERMEDIATE_DISTILL_EPOCHS:-$EPOCHS}
INTERMEDIATE_WEIGHT=${INTERMEDIATE_WEIGHT:-0.2}
ENCODER_DISTILL_LR=${ENCODER_DISTILL_LR:-1e-4}

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [ "$DEVICE" = "cuda" ]; then
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
    f"memory={properties.total_memory / 1024**3:.1f} GB"
)
PY
fi

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

if [ -z "$SPLIT_MANIFEST" ]; then
  SPLIT_MANIFEST="$RUN_DIR/calibration_split.json"
fi
if [ ! -f "$SPLIT_MANIFEST" ]; then
  run_benchmark "$PYTHON_BIN" make_calibration_split.py \
    --data_root "$DATA_ROOT" --out "$SPLIT_MANIFEST" \
    --fraction "$CALIBRATION_FRACTION"
else
  echo "[benchmark] reusing split manifest: $SPLIT_MANIFEST"
fi

COMMON=(--data_root "$DATA_ROOT" --out_dir "$RUN_DIR" --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" --workers "$WORKERS" --device "$DEVICE" \
        --cpu_threads "$CPU_THREADS" --gpu_memory_fraction "$GPU_MEMORY_FRACTION" \
        --image_normalization "$IMAGE_NORMALIZATION" --val_every "$VAL_EVERY" \
        --lr "$LR" --split_manifest "$SPLIT_MANIFEST" --seed "$SEED" \
        "${AMP_ARGS[@]}")

echo "[benchmark] device=$DEVICE normalization=$IMAGE_NORMALIZATION batch=$BATCH_SIZE workers=$WORKERS cpu_threads=$CPU_THREADS amp=$AMP val_every=$VAL_EVERY"
echo "[benchmark] tasks are strictly sequential; output=$RUN_DIR"

run_benchmark "$PYTHON_BIN" train.py --mode stl --task segmentation "${COMMON[@]}"
run_benchmark "$PYTHON_BIN" train.py --mode stl --task depth        "${COMMON[@]}"
run_benchmark "$PYTHON_BIN" train.py --mode stl --task normal       "${COMMON[@]}"
run_benchmark "$PYTHON_BIN" train.py --mode mtl                     "${COMMON[@]}"

run_benchmark "$PYTHON_BIN" merge_average.py --run_dir "$RUN_DIR" --out "$RUN_DIR/merged_average.pth"
if [ "$RUN_HEAD_DISTILLATION" = "1" ]; then
  run_benchmark "$PYTHON_BIN" distill_heads.py \
    --data_root "$DATA_ROOT" --run_dir "$RUN_DIR" \
    --split_manifest "$SPLIT_MANIFEST" \
    --merged "$RUN_DIR/merged_average.pth" \
    --out "$RUN_DIR/merged_distilled.pth" \
    --epochs "$DISTILL_EPOCHS" --batch_size "$DISTILL_BATCH_SIZE" \
    --lr "$DISTILL_LR" --val_every "$DISTILL_VAL_EVERY" \
    --bn_calibration_passes "$BN_CALIBRATION_PASSES" \
    --device "$DEVICE" --cpu_threads "$CPU_THREADS" \
    --gpu_memory_fraction "$GPU_MEMORY_FRACTION" \
    --image_normalization "$IMAGE_NORMALIZATION" "${AMP_ARGS[@]}"
fi
run_benchmark "$PYTHON_BIN" distill_heads.py \
  --data_root "$DATA_ROOT" --run_dir "$RUN_DIR" \
  --split_manifest "$SPLIT_MANIFEST" \
  --merged "$RUN_DIR/merged_average.pth" \
  --out "$RUN_DIR/merged_output_distilled.pth" \
  --epochs "$OUTPUT_DISTILL_EPOCHS" \
  --batch_size "$DISTILL_BATCH_SIZE" --lr "$OUTPUT_DISTILL_LR" \
  --train_scope heads --no-recalibrate_bn \
  --seg_sup_weight 0 --depth_sup_weight 0 --normal_sup_weight 0 \
  --method_name encoder_weight_average_output_distillation \
  --val_every "$DISTILL_VAL_EVERY" \
  --device "$DEVICE" --cpu_threads "$CPU_THREADS" \
  --gpu_memory_fraction "$GPU_MEMORY_FRACTION" \
  --image_normalization "$IMAGE_NORMALIZATION" "${AMP_ARGS[@]}"
run_benchmark "$PYTHON_BIN" distill_heads.py \
  --data_root "$DATA_ROOT" --run_dir "$RUN_DIR" \
  --split_manifest "$SPLIT_MANIFEST" \
  --merged "$RUN_DIR/merged_average.pth" \
  --out "$RUN_DIR/merged_intermediate_distilled.pth" \
  --epochs "$INTERMEDIATE_DISTILL_EPOCHS" \
  --batch_size "$DISTILL_BATCH_SIZE" --lr "$DISTILL_LR" \
  --train_scope encoder_heads --encoder_layers layer3,layer4 \
  --encoder_lr "$ENCODER_DISTILL_LR" \
  --intermediate_weight "$INTERMEDIATE_WEIGHT" \
  --val_every "$DISTILL_VAL_EVERY" \
  --bn_calibration_passes "$BN_CALIBRATION_PASSES" \
  --device "$DEVICE" --cpu_threads "$CPU_THREADS" \
  --gpu_memory_fraction "$GPU_MEMORY_FRACTION" \
  --image_normalization "$IMAGE_NORMALIZATION" "${AMP_ARGS[@]}"
run_benchmark "$PYTHON_BIN" evaluate.py --data_root "$DATA_ROOT" --run_dir "$RUN_DIR" \
  --merged "$RUN_DIR/merged_average.pth" --csv "$RUN_DIR/benchmark_results.csv" \
  --output_distilled "$RUN_DIR/merged_output_distilled.pth" \
  --intermediate_distilled "$RUN_DIR/merged_intermediate_distilled.pth" \
  --batch_size "$EVAL_BATCH_SIZE" --workers "$WORKERS" --device "$DEVICE" \
  --cpu_threads "$CPU_THREADS" --gpu_memory_fraction "$GPU_MEMORY_FRACTION" \
  --image_normalization "$IMAGE_NORMALIZATION" --seed "$SEED" \
  "${AMP_ARGS[@]}"

echo "Benchmark completed: $RUN_DIR/benchmark_results.csv"
