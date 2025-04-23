# scripts/train.py

# 训练模块 / トレーニングモジュール / Training Module
import torch
class Trainer:
    def __init__(self, model, train_loader, test_loader, device, criterion, optimizer):
        """
        初始化训练器 / トレーナーを初期化 / Initialize the trainer
        :param model: 神经网络模型 / ニューラルネットワークモデル / Neural network model
        :param train_loader: 训练数据加载器 / トレーニングデータローダー / Training data loader
        :param test_loader: 测试数据加载器 / テストデータローダー / Test data loader
        :param device: 运行设备 (CPU/GPU) / 実行デバイス (CPU/GPU) / Execution device (CPU/GPU)
        :param criterion: 损失函数 / 損失関数 / Loss function
        :param optimizer: 优化器 / オプティマイザー / Optimizer
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.input_shape = None  # 用于存储模型期望的输入形状

    def _check_input_shape(self, images):
        """
        检查输入图片的形状是否与模型期望的形状一致。
        """
        if self.input_shape is None:
            self.input_shape = images.shape[1:]  # 初始化模型期望的输入形状
        if images.shape[1:] != self.input_shape:
            raise ValueError(f"输入图片的形状 {images.shape[1:]} 与模型期望的形状 {self.input_shape} 不匹配！")

    def train(self, epochs):
        """
        训练模型 / モデルをトレーニング / Train the model
        :param epochs: 训练的轮数 / トレーニングのエポック数 / Number of training epochs
        """
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                print("input shape:", images.shape)  # 打印输入图片shape
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self._check_input_shape(images)  # 检查输入形状

                    # 前向传播
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                except Exception as e:
                    print(f"训练过程中发生错误: {e}")

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def test(self, test_loader=None):
        """
        测试模型，支持自定义测试集。
        :param test_loader: 可选，传入自定义的DataLoader作为测试集。
        """
        self.model.eval()
        correct = 0
        total = 0
        loader = test_loader if test_loader is not None else self.test_loader

        with torch.no_grad():
            for images, labels in loader:
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self._check_input_shape(images)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                except Exception as e:
                    print(f"测试过程中发生错误: {e}")

        accuracy = correct / total if total > 0 else 0
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def calculate_accuracy(self):
        """
        计算模型在测试集上的准确率 / モデルのテストセットでの精度を計算 / Calculate model accuracy on the test set
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self._check_input_shape(images)  # 检查输入形状

                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                except Exception as e:
                    print(f"计算准确率过程中发生错误: {e}")

        accuracy = correct / total if total > 0 else 0
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy