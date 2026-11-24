# Encoder Averaging + Intermediate Feature + Output Distillation

Date: 2026-11-19

## Method

This experiment adds a second fusion method without replacing the existing
head-only distillation baseline:

1. Average the three STL encoders.
2. Recalibrate encoder BatchNorm on the held-out calibration split.
3. Freeze stem, layer1, and layer2.
4. Unfreeze encoder `layer3` and `layer4` with a lower learning rate.
5. Train the three expert task heads.
6. Distill intermediate layer3/layer4 features from each task's corresponding
   STL teacher.
7. Distill final segmentation, depth, and normal outputs from those teachers.

The intermediate loss uses normalized feature cosine distance at layer3 and
layer4:

```text
L_feature = mean(1 - cosine(student_feature, task_teacher_feature))
```

The output losses are the same task-specific KD losses used by the head-only
method. The optimized objective is:

```text
L = mean(segmentation, depth, normal output losses)
  + 0.2 * L_feature
```

## Data

- Source: `/Users/dell/Projects/nyuv2`
- Teacher/MTL train pool: 715 images
- Calibration split: 80 images
- Validation: 654 images
- Calibration and teacher training overlap: 0

## Configuration

| Parameter | Value |
|---|---:|
| Teacher/MTL epochs | 5 |
| Base batch size | 4 |
| Base optimizer | Adam |
| Base learning rate | 2e-4 |
| Distillation epochs | 10 |
| Distillation batch size | 4 |
| Trainable encoder layers | layer3, layer4 |
| Head learning rate | 1e-3 |
| Encoder learning rate | 1e-4 |
| Optimizer | AdamW |
| Intermediate loss weight | 0.2 |
| Segmentation temperature | 2 |
| Input normalization | ImageNet |
| Device | Apple M5 MPS, FP16 autocast |

## Results

| Method | Seg mIoU | Seg pixel acc. | Depth abs. err. | Depth rel. err. | Normal mean | Normal median |
|---|---:|---:|---:|---:|---:|---:|
| STL experts | **49.42%** | **71.93%** | **0.465** | **0.190** | **28.38 deg** | **23.07 deg** |
| MTL | 48.23% | 71.11% | 0.452 | 0.185 | 29.21 deg | 24.07 deg |
| Encoder averaging | 44.25% | 68.75% | 0.532 | 0.218 | 31.10 deg | 26.87 deg |
| Encoder averaging + BN calibration | 44.93% | 68.72% | 0.541 | 0.232 | 31.44 deg | 26.78 deg |
| Encoder averaging + BN + head KD | 44.51% | 68.29% | **0.519** | **0.219** | 30.73 deg | 26.70 deg |
| Encoder averaging + BN + IFD + OD | **45.50%** | **69.07%** | 0.523 | 0.225 | **30.57 deg** | **26.61 deg** |

## Comparison

Intermediate + output distillation versus head-only distillation:

- Segmentation mIoU: 44.51% -> 45.50%, +0.98 percentage points.
- Segmentation pixel accuracy: 68.29% -> 69.07%, +0.78 percentage points.
- Depth absolute error: 0.519 -> 0.523, slightly worse.
- Depth relative error: 0.219 -> 0.225, worse.
- Normal mean angular error: 30.73 -> 30.57 degrees, improved by 0.16 degrees.

Against plain encoder averaging, the new method:

- improves segmentation mIoU by 1.24 percentage points;
- improves depth absolute error by about 1.5%;
- worsens depth relative error from 0.218 to 0.225;
- improves normal mean angular error by 0.53 degrees.

## Conclusion

Intermediate-feature distillation is a useful direction:

- It is the best fused method for segmentation and normal estimation.
- It closes part of the gap left by simple encoder averaging.
- It has a small depth tradeoff compared with head-only distillation.
- It still remains below STL and MTL on all primary metrics.

The shared student encoder cannot exactly match three different teacher feature
spaces. This limits how much intermediate distillation can recover. A natural
next experiment is to add a small per-task feature adapter before matching each
teacher, while keeping the averaged encoder shared.

## Artifacts

- Comparison CSV: `runs_full_ds5_seed0/benchmark_results.csv`
- Intermediate/output distilled model:
  `runs_full_ds5_seed0/merged_intermediate_distilled.pth`
- Calibration split: `splits/nyuv2_seed0_10pct.json`
