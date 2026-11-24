# Partial Three-Seed 100-Epoch Run

Date: 2026-11-24

The experiment was stopped at the user's request. This report preserves the
completed work only; it is not a final five-method result.

## Configuration

- Random seeds: `8394`, `4197`, `9424`
- Planned epoch budget: 100 for every trained method
- Teacher/MTL data: 715 images
- Calibration data: 80 images
- Validation data: 654 images
- Batch size: 8
- Base learning rate: `4e-4`
- Device: Apple M5 MPS with FP16 autocast
- Distillation protocol: output KD and intermediate-feature KD without
  calibration labels or BN recalibration

## Preserved Results

| Seed | Method | Progress | Best checkpoint | Seg mIoU | Seg Pix Acc | Depth Abs | Depth Rel |
|---:|---|---|---:|---:|---:|---:|---:|
| 8394 | STL segmentation | 100/100 | epoch 10 | 50.36% | 73.28% | - | - |
| 8394 | STL depth | 81/100 | epoch 80 | - | - | 0.4079 | 0.1724 |

MTL, encoder averaging, output distillation, intermediate-feature/output
distillation, and normal-estimation STL did not complete. Consequently, no
cross-seed mean, standard deviation, or final delta m value is reported.

## Local Artifacts

- Run root: `runs_three_seed_100ep`
- Segmentation checkpoint: `seed_8394/stl_segmentation/best.pth`
- Depth checkpoint: `seed_8394/stl_depth/best.pth`
- Stage logs and completion markers are retained under the same seed directory.

Large checkpoints and logs remain local and are intentionally excluded from the
Git repository.
