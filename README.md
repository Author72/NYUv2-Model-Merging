# NYUv2 Model Merging Benchmark (STL / MTL / Encoder Weight Averaging)

This is a compact PyTorch benchmark for three NYUv2 tasks:

- semantic segmentation (13 classes)
- depth estimation
- surface normal estimation

It implements:
1. Three single-task experts (STL).
2. One shared-encoder multi-task model (MTL).
3. Direct arithmetic averaging of the three STL encoders, while retaining each expert's task-specific head.
4. Optional task-head distillation on a held-out calibration split.
5. Encoder averaging plus intermediate-feature and output distillation.
6. Encoder averaging plus pure teacher-output distillation.

## 1. Dataset

This code expects the preprocessed NYUv2 dataset format used by LibMTL/MTAN:

```text
nyuv2/
├── train/
│   ├── image/    0.npy, 1.npy, ...
│   ├── label/    0.npy, 1.npy, ...
│   ├── depth/    0.npy, 1.npy, ...
│   └── normal/   0.npy, 1.npy, ...
└── val/
    ├── image/
    ├── label/
    ├── depth/
    └── normal/
```

The commonly used preprocessed benchmark has 795 train and 654 validation images, RGB inputs at 3x288x384, 13-class semantic labels, depth maps, and surface normals.

Download the preprocessed data linked from the LibMTL NYUv2 documentation:
https://libmtl.readthedocs.io/en/latest/docs/user_guide/benchmark/nyuv2.html

Do not mix this preprocessed format with the raw NYUv2 `.mat` files. The loader here is for the preprocessed `*.npy` version.

## 2. Environment

Recommended: a recent arm64 Python (3.10 or 3.11) and a current
PyTorch/torchvision pair. On Apple Silicon, install the normal arm64 PyTorch
wheel; a separate CUDA wheel is not used.

```bash
conda create -n nyu_merge python=3.10 -y
conda activate nyu_merge
pip install -r requirements-macos.txt
```

Check the active interpreter and Apple Metal support:

```bash
python -c "import platform, torch; print('arch=', platform.machine()); print('torch=', torch.__version__); print('mps_built=', torch.backends.mps.is_built()); print('mps_available=', torch.backends.mps.is_available())"
```

`arch` must be `arm64` and `mps_available` should be `True` on an M-series Mac.
If `arch` is `x86_64`, the process is running under Rosetta; recreate the
environment with a native arm64 Python. This benchmark is tested with PyTorch
2.12 on macOS 26 / Apple M5.

The first run downloads ImageNet-pretrained ResNet-50 weights if they are not already cached. If the machine has no network access, use `--no-pretrained` in every training command; the script will create and reuse one common random encoder initialization in `runs/init_encoder.pth`.

### Apple Silicon device behavior

- `--device auto` prefers Metal (`mps`), then CUDA, then CPU.
- Explicit `--device mps` validates MPS and fails with a clear message when the
  installed PyTorch cannot use it.
- Explicit `--device cpu` is still available for debugging.
- Unsupported MPS operators fall back to CPU through
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- MPS training uses FP16 autocast by default while masked losses remain FP32.
- The M5 defaults are batch size 4, learning rate 2e-4, zero DataLoader workers, six PyTorch CPU
  threads, and validation every two epochs. On macOS, persistent workers cost
  more to start and stop than they save for this `.npy` dataset.
- CPU usage may remain low while MPS is saturated. Use Activity Monitor's GPU
  history or `powermetrics` to inspect Metal utilization rather than treating
  idle CPU percentage as unused capacity.
- Validation predictions are copied to CPU before masked loss and metric
  calculations. Network training and inference still run on MPS; this avoids a
  Metal target-transfer race and keeps evaluation deterministic across devices.

## 3. Quick smoke test

Before a full run, confirm the data path and MPS path with one epoch, batch size
1, and no DataLoader workers:

```bash
python train.py --mode stl --task segmentation \
  --data_root /path/to/nyuv2 --out_dir ./runs_smoke \
  --epochs 1 --batch_size 4 --workers 0 --device mps --cpu_threads 6 --amp
```

The startup log should contain `Apple Metal (MPS)`. Use `--device cpu` if MPS is
unavailable. The default command uses `--device auto`, batch size 4,
learning rate 2e-4, zero DataLoader workers, six PyTorch CPU threads, and AMP
on MPS.

## 4. Run the complete first benchmark

### Five methods, three random seeds, 100 epochs

For the final five-method comparison, use the resumable multi-seed runner.
It trains STL, MTL, encoder averaging, encoder averaging plus output
distillation, and encoder averaging plus layer3/layer4 intermediate-feature
and output distillation. All trained methods use the same 100-epoch budget,
the same 10% calibration split, and the same ImageNet-pretrained encoder
initialization.

| Reported method | Implementation |
|---|---|
| STL | `train.py --mode stl --task {segmentation,depth,normal}` |
| MTL | `train.py --mode mtl` |
| EncoderAvg | `merge_average.py` |
| EncoderAvg+OutputKD | `distill_heads.py --train_scope heads` with teacher output losses |
| EncoderAvg+IntermediateKD+OutputKD | `distill_heads.py --train_scope encoder_heads --encoder_layers layer3,layer4` with layer3/layer4 feature losses plus teacher output losses |

```bash
/Users/dell/miniconda3/envs/cellpose/bin/python \
  run_three_seed_benchmark.py \
  --data_root /Users/dell/Projects/nyuv2 \
  --output_root ./runs_three_seed_100ep \
  --seeds 8394,4197,9424 \
  --init_encoder ./runs_full_ds5_seed0/init_encoder.pth \
  --epochs 100 --distill_epochs 100 \
  --batch_size 8 --lr 4e-4 --device mps
```

The seeds in this command were randomly sampled from `1..9999` with
`secrets.SystemRandom`. Each seed is written to its own subdirectory. Stage
logs and completion markers allow an interrupted suite to resume without
retraining completed stages.

The two distillation methods use calibration images but no calibration labels.
They do not recalibrate BN statistics, which keeps the comparison focused on
teacher-output and intermediate-feature distillation. The intermediate method
unfreezes only `layer3` and `layer4`.

The final files are:

```text
runs_three_seed_100ep/
├── experiment_manifest.json
├── all_seed_results.csv
├── aggregate_results.csv
├── aggregate_results.json
├── three_seed_report.md
└── seed_<seed>/
    ├── benchmark_results.csv
    ├── benchmark_task_results.csv
    ├── logs/
    └── .completed/
```

`benchmark_results.csv` contains one row per method with Seg mIoU, Seg Pix Acc,
Depth Abs, Depth Rel, Normal Mean, Normal Median, and Δm%. The cross-seed
report contains mean ± sample standard deviation.

### Single-run benchmark

**Apple Silicon mode (recommended):** defaults to automatic MPS selection, FP16
autocast, batch size 4, learning rate 2e-4, zero DataLoader workers, six
PyTorch CPU threads, normal process priority, validation every two epochs, and
strictly sequential tasks. Evaluation uses batch size 8. A portable lock
prevents duplicate runs.

```bash
bash run_first_benchmark.sh /absolute/path/to/nyuv2 ./runs_m5
```

The M5 has 32 GB unified memory shared by the CPU and GPU. Batch size 4 used
about 2.3 GB of MPS driver memory for STL and 3.4 GB for MTL in the local
micro-benchmark. If throughput matters more than convergence at a fixed epoch
count, try a larger batch with a linearly scaled learning rate:

```bash
BATCH_SIZE=8 LR=4e-4 DEVICE=mps bash run_first_benchmark.sh \
  /absolute/path/to/nyuv2 ./runs_m5
```

For a slower low-impact CPU run:

```bash
DEVICE=cpu CPU_THREADS=4 WORKERS=0 BATCH_SIZE=2 \
  bash run_first_benchmark.sh /absolute/path/to/nyuv2 ./runs_cpu
```

On a CUDA host, CUDA remains supported. The allocator cap defaults to 25% of
total GPU memory:

```bash
DEVICE=cuda GPU_MEMORY_FRACTION=0.20 \
  bash run_first_benchmark.sh /absolute/path/to/nyuv2 ./runs_cuda
```

The script performs, in order:
- Create or reuse a deterministic 10% calibration split from the training data.
- STL segmentation training
- STL depth training
- STL normal training
- MTL training
- encoder arithmetic averaging
- optional encoder BatchNorm calibration and three-head distillation
- pure teacher-output distillation without calibration labels
- layer3/layer4 intermediate-feature distillation plus output distillation
- validation evaluation and CSV export

Outputs:

```text
runs/
├── init_encoder.pth
├── stl_segmentation/best.pth
├── stl_depth/best.pth
├── stl_normal/best.pth
├── mtl/best.pth
├── merged_average.pth
├── merged_distilled.pth
├── merged_output_distilled.pth
├── merged_intermediate_distilled.pth
├── calibration_split.json
└── benchmark_results.csv
```

The calibration images are excluded from all teacher and MTL training runs.
Head distillation freezes the averaged encoder and optimizes only the three
expert heads using:

- segmentation teacher logits plus supervised cross entropy;
- depth teacher Smooth-L1 plus masked supervised loss;
- normal teacher cosine loss plus masked supervised loss.

The intermediate-feature method starts from the same averaged encoder, unfreezes
only `layer3` and `layer4`, and additionally aligns normalized layer3/layer4
features with each task's STL teacher:

```text
L = mean(output KD tasks)
  + intermediate_weight * mean_feature_cosine_loss(layer3, layer4)
```

The pure output-distillation method keeps the averaged encoder and BN statistics
fixed, trains only task heads, and does not use calibration labels. It is useful
when teacher predictions are available but labels are limited.

Useful environment variables:

```bash
CALIBRATION_FRACTION=0.10 \
DISTILL_EPOCHS=100 \
DISTILL_LR=1e-3 \
DISTILL_BATCH_SIZE=4 \
  bash run_first_benchmark.sh /absolute/path/to/nyuv2 ./runs_m5
```

The same script also runs the intermediate-feature variant. Relevant controls:

```bash
INTERMEDIATE_DISTILL_EPOCHS=100
INTERMEDIATE_WEIGHT=0.2
ENCODER_DISTILL_LR=1e-4
OUTPUT_DISTILL_EPOCHS=100
OUTPUT_DISTILL_LR=1e-3
```

Generate a split manually:

```bash
python make_calibration_split.py \
  --data_root /absolute/path/to/nyuv2 \
  --out ./splits/nyuv2_seed0_10pct.json \
  --fraction 0.10
```

Run only head distillation after teachers and `merged_average.pth` exist:

```bash
python distill_heads.py \
  --data_root /absolute/path/to/nyuv2 \
  --run_dir ./runs \
  --split_manifest ./splits/nyuv2_seed0_10pct.json \
  --merged ./runs/merged_average.pth \
  --out ./runs/merged_distilled.pth \
  --epochs 10 --batch_size 4 --lr 1e-3
```

Resume behavior: rerunning the script currently restarts training and overwrites checkpoints. It reuses the common initial encoder checkpoint. Copy the `runs` directory before a new seed/setting.

## 5. Run individual commands

Train STL segmentation:

```bash
python train.py --mode stl --task segmentation --data_root /path/to/nyuv2 --out_dir ./runs --epochs 100 --device mps
```

Train STL depth:

```bash
python train.py --mode stl --task depth --data_root /path/to/nyuv2 --out_dir ./runs --epochs 100 --device mps
```

Train STL normal:

```bash
python train.py --mode stl --task normal --data_root /path/to/nyuv2 --out_dir ./runs --epochs 100 --device mps
```

Train MTL:

```bash
python train.py --mode mtl --data_root /path/to/nyuv2 --out_dir ./runs --epochs 100 --device mps
```

Merge and evaluate:

```bash
python merge_average.py --run_dir ./runs --out ./runs/merged_average.pth
python evaluate.py --data_root /path/to/nyuv2 --run_dir ./runs \
  --merged ./runs/merged_average.pth --csv ./runs/benchmark_results.csv
```

## 6. Important implementation details

- The encoder is torchvision ResNet-50 with output stride 16 (dilated final stage).
- Each task has its own ASPP-based head.
- All runs reuse `runs/init_encoder.pth`, so STL and MTL start from the same encoder weights.
- The STL experts are trained independently after initialization.
- MTL optimizes the arithmetic mean of the three task losses.
- MPS uses FP16 autocast; segmentation/depth/normal reductions remain FP32.
- Batch 4 uses learning rate 2e-4, scaled linearly from 1e-4 at batch 2.
- Validation runs on epochs 1, even epochs, and the final epoch by default.
- Only `best.pth` is written by default. Use `--save_last` if `last.pth` is
  needed for external tooling.
- Segmentation uses cross entropy with `ignore_index=-1`.
- Depth uses masked L1 loss; zero depth is treated as invalid.
- Normal uses masked cosine loss; zero normal vectors are treated as invalid.
- Evaluation reports mIoU / pixel accuracy, depth absolute / relative error, and normal angular metrics.
- The merged model averages floating-point encoder state entries (including BatchNorm running statistics). Integer buffers such as `num_batches_tracked` use elementwise maximum. Each task head is copied from its corresponding STL expert.
- Head distillation optionally recalibrates encoder BatchNorm statistics on the
  calibration split, then freezes the encoder and optimizes only task heads.
- Intermediate/output distillation unfreezes only encoder layer3/layer4 at a
  lower encoder learning rate.
- `evaluate.py --distilled ...` includes optional task-head KD in the
  per-task diagnostic CSV.
- `evaluate.py --output_distilled ...` adds `EncoderAvg+OutputKD`.
- `evaluate.py --intermediate_distilled ...` adds
  `EncoderAvg+IntermediateKD+OutputKD`.

## 7. Research caveats (read before reporting results)

This is a **reproducible first baseline implementation**, not a claim of exact numerical reproduction of every LibMTL paper result.

1. The official LibMTL NYUv2 implementation uses its own dilated ResNet-50, ASPP decoder, loss/metric code, and benchmark training configuration. This project follows the same preprocessed data layout and task definitions but uses a compact standalone implementation.
2. The simple MTL objective is the unweighted mean of three losses. For a paper-quality comparison, consider also reproducing the official LibMTL configuration (e.g. EW weighting) and report the exact weighting method.
3. BatchNorm running statistics are buffers, not trainable parameters. Averaging them is a simple baseline choice, but it is not statistically exact. Add a separate ablation later: average trainable parameters only and recalibrate BatchNorm statistics on training images.
4. This first implementation keeps task-specific heads from their own experts. Therefore the method is accurately named **Encoder Weight Averaging + Expert Heads**, not full-model parameter averaging.
5. Run at least three random seeds for final results. Keep data split, augmentation, optimizer, epoch budget, and evaluation code identical across methods.
6. Do not use validation labels for training. The `val` split is used only for model selection and reporting.

## 8. Metrics

| Task | Metrics | Better |
|---|---|---|
| Segmentation | mIoU, pixel accuracy | higher |
| Depth | absolute error, relative error | lower |
| Normal | mean / median angular error | lower |
| Normal | accuracy below 11.25 / 22.5 / 30 degrees | higher |

The main CSV stores one row per method. For STL, each metric is evaluated using
the corresponding task specialist. For MTL and all fused methods, all three
tasks are evaluated from the same multi-head model.

Δm% is the mean relative improvement over STL across the six reported metrics:

```text
higher-is-better metric: (method - STL) / |STL| * 100
lower-is-better metric:  (STL - method) / |STL| * 100
Δm% = mean(six normalized metric improvements)
```

Thus Δm% is positive when a method improves over STL on average across the six
metrics. The per-task diagnostic CSV retains the broader set of auxiliary
loss and threshold-accuracy metrics.
