# Encoder Averaging + Pure Output Distillation

Date: 2026-11-19

## Method

This is a clean output-distillation baseline:

1. Average the three STL encoders.
2. Keep the averaged encoder fixed.
3. Do not recalibrate encoder BatchNorm.
4. Freeze all encoder parameters.
5. Optimize only the three expert task heads.
6. Do not use calibration ground-truth labels.
7. Distill only from the corresponding STL teachers:
   - segmentation logits with temperature 2;
   - depth outputs with Smooth-L1;
   - normal directions with cosine loss.

## Configuration

| Parameter | Value |
|---|---:|
| Calibration images | 80 |
| Distillation epochs | 10 |
| Batch size | 4 |
| Head optimizer | AdamW |
| Head learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Segmentation temperature | 2 |
| BN recalibration | Disabled |
| Trainable encoder layers | None |
| Ground-truth supervision | Disabled |
| Device | Apple M5 MPS, FP16 autocast |

## Results

| Method | Seg mIoU | Seg pixel acc. | Depth abs. err. | Depth rel. err. | Normal mean | Normal median |
|---|---:|---:|---:|---:|---:|---:|
| STL experts | **49.42%** | **71.93%** | 0.465 | **0.190** | **28.38 deg** | **23.07 deg** |
| MTL | 48.23% | 71.11% | **0.452** | 0.185 | 29.21 deg | 24.07 deg |
| Encoder averaging | 44.25% | 68.75% | 0.532 | 0.218 | 31.10 deg | 26.87 deg |
| Averaging + Head KD + supervision | 44.51% | 68.29% | **0.519** | 0.219 | 30.73 deg | 26.70 deg |
| Averaging + pure output distillation | **44.85%** | 68.61% | 0.521 | 0.220 | 30.75 deg | 26.84 deg |
| Averaging + intermediate + output distillation | **45.50%** | **69.07%** | 0.523 | 0.225 | **30.57 deg** | **26.61 deg** |

## Findings

Compared with plain encoder averaging, pure output distillation:

- improves segmentation mIoU by 0.60 percentage points;
- slightly reduces pixel accuracy by 0.14 percentage points;
- improves depth absolute error by about 2.0%;
- slightly worsens depth relative error;
- improves normal mean angular error by 0.34 degrees.

Compared with label-supervised head KD:

- pure output distillation is slightly better on segmentation;
- supervised head KD remains better on depth and normal.

Compared with intermediate + output distillation:

- intermediate distillation is clearly better on segmentation and normal;
- pure output distillation is marginally better on depth.

## Conclusion

Pure output distillation is valid and improves the plain averaged model on
segmentation, depth absolute error, and normal angle. It is useful when only
teacher predictions are available and no calibration labels can be used.

It still remains well below STL and MTL. Output distillation can adapt task
heads to the new feature distribution, but cannot recover encoder information
lost during averaging.

## Artifact

- Model: `runs_full_ds5_seed0/merged_output_distilled.pth`
- Six-method CSV: `runs_full_ds5_seed0/benchmark_results.csv`
