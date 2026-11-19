# Apple M5 Training Efficiency

Measured with PyTorch 2.12.0 on an Apple M5 with 32 GB unified memory.
The real-data tests use the same 100-image train / 131-image validation subset
as `experiments/micro_m5_3ep`.

## Short synthetic training-step benchmark

- Batch 2, FP32: 13.36 samples/s
- Batch 4, FP32: 14.63 samples/s
- Batch 4, FP16 autocast: 22.74 samples/s
- Batch 4, channels-last: unsupported stride error on MPS
- MTL batch 8, FP16 autocast: 17.60 samples/s
- MTL batch 16, FP16 autocast: 18.90 samples/s

## Real-data end-to-end results

- Optimized STL segmentation, batch 4 + AMP: about 19.1 training samples/s;
  3 epochs completed in 31.06 s including two validations.
- Optimized MTL, batch 4 + AMP: about 13.7 training samples/s;
  3 epochs completed in 51.50 s including two validations.
- The previous non-AMP batch-2 segmentation run was about 5.3 training
  samples/s, so the optimized path is roughly 3.6x faster in training.
- MPS driver memory was about 2.3 GB for STL and 3.4 GB for MTL at batch 4.

## Quality check

For segmentation at 3 epochs:

| Configuration | Seg mIoU | Seg pixel accuracy |
|---|---:|---:|
| Old batch 2, FP32, lr 1e-4 | 28.58% | 59.26% |
| Batch 4, AMP, lr 2e-4 | 26.78% | 58.59% |
| Batch 8, AMP, lr 4e-4 | 27.41% | 59.28% |

Batch 8 is slightly faster, but it has fewer optimizer updates per epoch. For
MTL, batch 8/4e-4 gave 3-epoch depth error 2.06 and normal error 41.87 degrees;
batch 4/2e-4 improved these to 1.57 and 40.32 degrees. The default therefore
uses batch 4/2e-4 and keeps batch 8 available for throughput-focused runs.

## Why CPU usage stays low

The convolution workload runs on Metal/GPU. CPU workers mostly load `.npy`
files, which is too light to occupy the M5 CPU. On macOS, persistent DataLoader
workers also add a long teardown cost. The optimized path therefore uses zero
workers by default, which reduced one-epoch wall time from 50.88 s to 9.98 s in
the local comparison.
