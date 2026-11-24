# Encoder Averaging + Expert-Head Distillation

Date: 2026-11-19

## Experiment summary

The averaged encoder was frozen and the three expert heads were distilled on a
small calibration set. This run verifies the implementation and measures
whether head adaptation can repair the mismatch between the averaged encoder
and the original expert heads.

## Data

- Source: `/Users/dell/Projects/nyuv2`
- Train pool: 715 images
- Calibration: 80 images, 10% of the original training split
- Validation: all 654 validation images
- Teacher/MTL training and calibration are disjoint.
- Split manifest: `splits/nyuv2_seed0_10pct.json`

## Shared training setup

- Encoder: ImageNet-pretrained ResNet-50 with stride 16
- Tasks: 13-class segmentation, depth, surface normals
- Input: ImageNet mean/std normalization
- Device: Apple M5 MPS with FP16 autocast
- Optimizer: Adam
- Batch size: 4
- Learning rate: 2e-4
- Weight decay: 1e-5
- Seed: 0
- Teacher/MTL epochs: 5
- Validation epochs: 1 and 5

## Head distillation setup

- Student encoder: average of the three STL encoders
- Encoder state: frozen
- BN calibration: one pass over the 80 calibration images
- Trainable parameters: three expert task heads only
- Optimizer: AdamW
- Head learning rate: 1e-3
- Epochs: 10
- Batch size: 4
- Scheduler: cosine annealing
- Segmentation temperature: 2
- Segmentation supervised/KD weights: 1.0 / 1.0
- Depth teacher/supervised weights: 1.0 / 0.2
- Normal teacher/supervised weights: 1.0 / 0.2

## Results

| Method | Seg mIoU | Seg pixel acc. | Depth abs. err. | Depth rel. err. | Normal mean | Normal median |
|---|---:|---:|---:|---:|---:|---:|
| STL experts | **49.42%** | **71.93%** | 0.465 | **0.190** | **28.38 deg** | **23.07 deg** |
| MTL | 48.23% | 71.11% | **0.452** | 0.185 | 29.21 deg | 24.07 deg |
| Encoder averaging | 44.25% | 68.75% | 0.532 | 0.218 | 31.10 deg | 26.87 deg |
| Encoder averaging + BN calibration | 44.93% | 68.72% | 0.541 | 0.232 | 31.44 deg | 26.78 deg |
| Encoder averaging + BN + head KD | 44.51% | 68.29% | 0.519 | 0.219 | 30.73 deg | 26.70 deg |

### Comparison with the plain averaged model

Head distillation after BN calibration:

- Segmentation mIoU: 44.25% -> 44.51%, +0.26 percentage points.
- Segmentation pixel accuracy: 68.75% -> 68.29%, -0.46 percentage points.
- Depth absolute error: 0.532 -> 0.519, improved by about 2.5%.
- Depth relative error: 0.218 -> 0.219, slightly worse.
- Normal mean angular error: 31.10 -> 30.73 degrees, improved by 0.37 degrees.

### BN-only ablation

BN recalibration alone improves segmentation from 44.25% to 44.93% mIoU, but
degrades depth from 0.532 to 0.541 and normal from 31.10 to 31.44 degrees.

Head distillation then rebalances the model: segmentation decreases slightly
from the BN-only result, while depth and normal improve. The final model is a
compromise, not a simultaneous improvement over every starting point.

## Conclusion

The proposed modification is technically valid and runs successfully, but the
current result does not show a competitive merged model:

- STL remains the strongest overall method.
- MTL is close to STL and stronger on depth.
- Encoder averaging loses about 4.7 mIoU points relative to STL.
- BN calibration and head distillation recover only a small part of this gap.
- Head distillation can adjust outputs to the new feature distribution, but it
  cannot recover information lost in the averaged encoder.

The current recommendation is to keep this as a fusion baseline and continue
with multiple seeds and stronger merging methods. If this approach is retained,
the next ablation should compare teacher-only KD against KD with labels and test
unfreezing the final encoder stage.

## Artifacts

- Results CSV: `runs_full_ds5_seed0/benchmark_results.csv`
- Average merged model: `runs_full_ds5_seed0/merged_average.pth`
- BN-calibrated ablation: `runs_full_ds5_seed0/merged_bn_calibrated.pth`
- Distilled model: `runs_full_ds5_seed0/merged_distilled.pth`
- Calibration split: `splits/nyuv2_seed0_10pct.json`
