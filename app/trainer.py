# scripts/train.py

# 训练模块 / トレーニングモジュール / Training Module
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

    def train(self, epochs):
        """
        训练模型 / モデルをトレーニング / Train the model
        :param epochs: 训练的轮数 / トレーニングのエポック数 / Number of training epochs
        """
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def test(self):
        """
        测试模型 / モデルをテスト / Test the model
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")