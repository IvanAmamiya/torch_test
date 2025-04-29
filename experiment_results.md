# 神经网络实验记录 (2025-04-29)

## 实验目标
提升 CIFAR-10 数据集上的图像分类精度。

## 主要调整

1.  **模型结构优化**:
    *   初始模型为简单的 CNN。
    *   尝试删除/恢复局部池化层 (`nn.MaxPool2d`)。
    *   尝试添加/删除全局平均池化层 (`nn.AdaptiveAvgPool2d`)。
    *   增加卷积层数，提升模型深度。
    *   最终升级为类似 VGG 的结构，包含多个卷积块（Conv+BN+ReLU+MaxPool）。
    *   调整全连接层以匹配特征图尺寸变化。

2.  **正则化**:
    *   在卷积块后和全连接层中添加 `nn.Dropout` 层。
    *   在卷积块后添加 `nn.LocalResponseNorm` (LRN) 层。
    *   模型中广泛使用 `nn.BatchNorm2d`。

3.  **数据增强**:
    *   基础增强：Resize, RandomHorizontalFlip, RandomCrop, ColorJitter, Normalize。
    *   增加 Mixup 数据增强。
    *   增加 RandomErasing 数据增强。

4.  **训练机制**:
    *   确认使用了学习率递减机制 (如 `CosineAnnealingLR`)。
    *   清理了旧的模型权重 (`model.pth`) 以便从头训练。
    *   (曾尝试) 删除 checkpoint 机制，仅在最优时保存模型。

## 预期效果
*   更深、更复杂的模型结构有望提升特征提取能力和最终精度。
*   多种正则化手段（Dropout, LRN, BN）和数据增强（Mixup, RandomErasing）有助于提升模型的泛化能力，防止过拟合。
*   学习率递减有助于模型在训练后期更好地收敛。

## 后续建议
*   进行充分的训练，观察损失和精度曲线。
*   根据训练结果调整超参数（学习率、Dropout比例、优化器参数等）。
*   可以尝试更先进的模型结构（如 ResNet）或数据增强策略（如 AutoAugment）。

## 随机擦除 精度降低2% 83-81
Mixup 提升2%
