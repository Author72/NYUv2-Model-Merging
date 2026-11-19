# NYUv2 多任务模型融合实验阶段汇报

日期：2026-11-12

## 结论摘要

当前“编码器权重平均 + 保留各任务专家头”的方法在工程实现上可行，模型可以正常训练、合并和评估。但在目前的短周期小样本实验中，它没有稳定优于 STL 或 MTL：

- 分割：明显低于 STL。
- 深度：优于 STL 和 MTL。
- 法向量：与 STL、MTL 基本持平。

因此，当前结果适合作为 pipeline 和初始 baseline，还不足以证明该方法整体优于多任务联合训练或单任务专家。下一步需要全量数据、相同训练预算和多个随机种子验证。

## 数据结构与模型

- 数据集：LibMTL/MTAN 预处理 NYUv2。
- 全量数据：795 张训练图、654 张验证图。
- 本次对比子集：每 8 张取 1 张训练图，共 100 张；每 5 张取 1 张验证图，共 131 张。
- 输入：3 x 288 x 384 RGB。
- 任务：13 类语义分割、深度估计、表面法向量估计。
- 编码器：ImageNet 预训练 ResNet-50，最后一阶段使用空洞卷积，输出 stride 为 16。
- 任务头：每项任务独立 ASPP head。
- 公共初始化：三个 STL 和 MTL 使用同一个编码器初始权重。

## 主对比实验配置

主对比使用相同子集、相同初始化和相同训练预算：

| 参数 | 值 |
|---|---|
| Epoch | 3 |
| Batch size | 2 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Seed | 0 |
| Device | Apple M5 MPS |
| Input normalization | 无，仅缩放到 [0, 1] |
| Checkpoint selection | 三任务平均验证损失最小 |

该表对应 `runs_micro_3ep`，是当前唯一一组完整、同条件的 STL / MTL / Weight Averaging 对比。

## 主对比结果

| Method | Seg mIoU | Seg pixel acc. | Depth abs. err. | Depth rel. err. | Normal mean | Normal median |
|---|---:|---:|---:|---:|---:|---:|
| STL | **24.13%** | **56.91%** | 1.729 | 0.650 | 40.46 deg | 37.34 deg |
| MTL | 23.68% | 56.68% | 1.835 | 0.692 | **39.99 deg** | **36.30 deg** |
| Weight Averaging | 19.80% | 53.47% | **1.658** | **0.621** | 40.66 deg | 37.35 deg |

法向量阈值准确率：

| Method | < 11.25 deg | < 22.5 deg | < 30 deg |
|---|---:|---:|---:|
| STL | 8.12% | 25.74% | 38.24% |
| MTL | **9.12%** | **27.83%** | **40.31%** |
| Weight Averaging | 7.97% | 25.52% | 38.19% |

验证集多数类像素准确率约为 25.29%，所以分割模型已经学到有效信息，而不是简单预测多数类。

## 结果解释

### 1. MTL 与 STL 接近

MTL 的分割 mIoU 只比 STL 低 0.45 个百分点，像素准确率低 0.23 个百分点；法向量的平均角误差还略低。说明共享编码器没有严重破坏三项任务。

但 MTL 的深度绝对误差比 STL 高约 6.1%，说明深度任务受到联合优化的影响。

### 2. 简单权重平均不是全面优势

编码器平均使深度绝对误差从 1.729 降到 1.658，改善约 4.1%；相对误差也有改善。

但分割 mIoU 从 24.13% 降到 19.80%，相对下降约 18%。这表明不同任务的编码器参数更新方向存在冲突，简单算术平均不一定位于一个对三个任务都好的参数区域。

法向量几乎不变，因此当前结果不能支持“权重平均同时保持所有任务性能”的结论。

### 3. 当前融合方法与标题范围

当前实现不是完整模型平均，而是：

1. 平均三个 STL 编码器。
2. 每个任务的 head 复制自对应 STL 专家。

所以更准确的名称是 **Encoder Weight Averaging + Expert Heads**。

## 输入标准化补充实验

后续加入 ImageNet mean/std 标准化，并完成 STL 的 10 epoch 训练：

| Task | Best epoch | Metric |
|---|---:|---:|
| Segmentation | 8 | mIoU 32.33%，pixel acc. 61.16% |
| Depth | 9 | abs. error 0.840，rel. error 0.306 |
| Normal | 10 | mean angle 36.94 deg |

MTL 的 10 epoch 实验在 epoch 3 后为了运行时优化而停止，最佳 checkpoint 为 epoch 3：

- Seg mIoU：28.20%
- Depth abs. error：1.313
- Normal mean angle：38.12 deg

该补充实验与主对比的训练时长不同，不能直接和主表的 Weight Averaging 结论混用。

## 运行效率

完成 M5 优化后：

- STL batch 4 + FP16 autocast：约 19.1 training samples/s。
- MTL batch 4 + FP16 autocast：约 13.7 training samples/s。
- 相对于最初 batch 2 + FP32 的分割训练，吞吐约提升 3.6 倍。
- batch 4 时 MPS driver memory：STL 约 2.3 GB，MTL 约 3.4 GB。
- 默认每 2 epoch 验证一次，并取消未使用的 `last.pth` 写入。

CPU 空闲率较高是正常的，卷积计算发生在 M5 GPU；数据读取不是当前主要瓶颈。

## 可行性判断

### 技术上可行

- 三个专家能够独立训练。
- 编码器可以按张量逐元素平均。
- 专家头可以与平均编码器重新组合。
- 合并后的模型能正常完成三任务推理和统一评估。

### 研究上尚未证明有效

当前只有 3 epoch、100 张训练图、单随机种子，不能据此判断方法优劣。尤其简单权重平均对分割产生了明显负面影响，因此当前不应声称“模型融合优于 STL/MTL”。

更准确的说法是：

> 编码器权重平均可以作为多任务模型融合的 baseline；初步结果显示它可能改善深度，但会损害分割，是否存在稳定的多目标收益仍需更严格的实验验证。

## 建议的下一阶段实验

1. 使用完整 NYUv2 数据，至少 3 个随机种子。
2. STL、MTL 和 Weight Averaging 使用相同 epoch、优化器和验证标准。
3. 固定 ImageNet 输入标准化和 MPS AMP 配置。
4. 加入独立测试集，避免只在验证集上选择和报告。
5. 做编码器平均消融：
   - 只平均 trainable parameters。
   - 同时平均 BatchNorm statistics。
   - 平均后重新估计 BatchNorm statistics。
6. 与更稳定的融合方法比较：
   - Task Arithmetic。
   - Fisher-weighted averaging。
   - Model soup / greedy soup。
7. 报告均值、标准差和置信区间，而不是只报告单次最佳结果。

## 可直接向导师汇报的一句话

> 已完成 NYUv2 三任务 STL、共享编码器 MTL 和编码器权重平均的可运行 baseline。小样本短训练中 MTL 接近 STL，权重平均改善了深度但明显损伤分割，法向量基本持平；方法实现可行，但当前证据不足以证明简单权重平均是稳定有效的多任务融合方案，下一步需要全量数据、相同预算和三随机种子验证。
