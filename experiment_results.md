# 实验总结（2025-05-02）

## 1. 实验目标
- 探索 Mixup 数据增强对 CIFAR-10 图像分类任务的影响。
- 对比不同 Mixup alpha 系数下模型的训练与测试表现。

## 2. 数据集
- CIFAR-10
- FashionMNIST（部分代码支持，但本次主实验为 CIFAR-10）

## 3. 模型结构
- 基于 VGGNet 改进的卷积神经网络（CNN）
- 结构：
  - 多个卷积块（Conv2d + BatchNorm2d + ReLU + MaxPool2d + Dropout）
  - 全连接层（Flatten + Linear + ReLU + Dropout + Linear）
  - Dropout 用于正则化，防止过拟合

## 4. 优化器与损失函数
- 优化器：Adam（每个实验独立初始化，lr=0.001）
- 学习率调度器：StepLR（step_size=5, gamma=0.1）
- 损失函数：CrossEntropyLoss

## 5. 数据增强
- Mixup（alpha 系数从 0.0 到 1.0，步长 0.2）
- alpha=0.0 时为无 Mixup（原始训练）
- 其余 alpha 依次为 0.2, 0.4, 0.6, 0.8, 1.0

## 6. 实验流程
- 对每个 alpha：
  1. 重新初始化模型和优化器
  2. 用对应 Mixup alpha 的 DataLoader 训练模型
  3. 记录每个 epoch 的训练精度和 loss
  4. 训练完成后，保存多条精度曲线和 loss 曲线（每条线对应一个 alpha）
- 所有实验结果可视化保存在 plots/ 目录下

## 7. 可视化
- 精度曲线图：每条线代表一个 alpha，横轴为 epoch，纵轴为 accuracy
- Loss 曲线图：每条线代表一个 alpha，横轴为 epoch，纵轴为 loss
- 文件示例：mixup_alpha_curve_xxxx.png、mixup_alpha_loss_curve_xxxx.png

## 8. 结论建议
- 通过对比不同 alpha 的曲线，可以直观分析 Mixup 强度对模型泛化能力的影响
- 推荐根据实验曲线选择最优 alpha 作为后续训练的默认参数

## 結論

MIXUP手法は精度向上に有効ですが、データ拡張の強度（alpha）が大きければ大きいほど必ずしも精度が向上するわけではありません。適切なalphaを選択することが重要です。

---

## 実験記録（2025年5月2日）

- CIFAR-10データセットを用いた画像分類タスク。
- VGG風CNNモデルを採用。
- 最適化手法：Adam（学習率0.001）、StepLR（ステップサイズ5、gamma=0.1）。
- 損失関数：CrossEntropyLoss。
- データ拡張：Mixup（alpha=0.0, 0.2, 0.4, 0.6, 0.8, 1.0）。
- 各alphaごとにモデル・オプティマイザを初期化し、40エポック学習。
- 各エポックごとに精度・損失を記録。
- 結果はplots/ディレクトリに精度・損失曲線として保存。
- Mixup強度による汎化性能の違いを可視化し、最適なalphaを検討。

---

## VGGモデル詳細構造（ソースコード抜粋）

```python
class VGGNet(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.LocalResponseNorm(5),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.45),  # 全結合層のDropout
            nn.Linear(512, num_classes)
        )
```

- 各畳み込みブロックの後にBatchNorm, ReLU, MaxPool, Dropoutを配置。
- Block1の最後にLRN（局所応答正規化）を追加。
- 全結合層前にFlatten、512ユニット、ReLU、Dropout(0.45)を挿入。
- Dropoutは過学習防止のため各所で活用。
- 出力層はクラス数（CIFAR-10なら10）に対応。

この構造により、VGG系の深い特徴抽出と正則化を両立し、Mixup等のデータ拡張と組み合わせて高い汎化性能を目指した。

---

## 改善点・今後の改良案

- モデルの正則化強化：各畳み込みブロックと全結合層にDropoutを追加し、過学習を抑制。
- LRN（局所応答正規化）をBlock1に導入し、汎化性能を向上。
- Mixupデータ拡張のalphaを複数パターンで比較し、最適な強度を探索。
- 学習率スケジューラ（StepLR）のステップサイズを5に調整し、より安定した収束を目指した。
- 各alphaごとにモデル・オプティマイザを完全初期化し、実験の独立性を確保。
- 精度・損失の推移を全alphaで可視化し、データ拡張の効果を直感的に比較可能に。
- 無Mixup（alpha=0.0）も基準線として同時に評価。
- FastAPIによる進捗ストリーミング機能を導入し、サーバー側でリアルタイムに学習状況を把握可能。

今後は、
- CutMixやAutoAugment等、他のデータ拡張手法との比較
- モデル構造のさらなる深層化やResNet系への拡張
- 学習率やバッチサイズ等のハイパーパラメータ最適化
- 実験自動化・再現性向上のためのスクリプト整備
なども検討したい。

---

## パラメーターAlphaと精密度の関係及びその原因
- Mixupのalpha値が小さい場合（0.0や0.2）、データ拡張の効果は限定的で、精度は通常の学習と大きく変わらない。
- alphaが大きくなると（0.6～1.0）、サンプル間の混合が強くなり、モデルの汎化性能が向上する場合もあるが、過度な混合はラベルノイズを増やし、逆に精度が低下することもある。
- 実験結果より、最適なalphaはタスクやモデルによって異なり、必ずしも大きいほど良いとは限らない。

## 元々で訓練したモデルを使うと、精密度を80％に上がる原因
- 事前学習済みモデルや以前の重みを再利用した場合、初期値が良いため学習が早く進み、精度が80％程度まで上がることが多い。
- ただし、データ分布やモデル構造が変わった場合、古い重みが最適でないこともあり、過学習や収束不良のリスクもある。
- そのため、実験ごとにモデル・オプティマイザを初期化し、条件を揃えることが重要。

## 高いLOSSの原因、改善策
- MixupやCutMixなどの強いデータ拡張を使うと、ラベルが混合されるため、損失（LOSS）が高くなりやすい。
- 特にalphaが大きい場合、ラベルノイズが増え、モデルが正しいラベルを学びにくくなる。
- 改善策：
  - alpha値を適切に調整し、過度な混合を避ける
  - 学習率やバッチサイズを見直す
  - モデルの正則化（Dropout, LRN, BatchNorm等）を強化
  - 早期終了（EarlyStopping）や学習率スケジューラを活用
