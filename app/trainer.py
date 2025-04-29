# scripts/train.py

# 训练模块 / トレーニングモジュール / Training Module
import torch
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import datetime

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
        # Add lists to store metrics
        self.train_losses = []
        self.val_accuracies = []
        self.val_recalls = []
        self.epoch_dates = []  # 新增：记录每个epoch的日期
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Store timestamp

    def _check_input_shape(self, images):
        """
        检查输入图片的形状是否与模型期望的形状一致。
        """
        if self.input_shape is None:
            self.input_shape = images.shape[1:]  # 初始化模型期望的输入形状
        if images.shape[1:] != self.input_shape:
            raise ValueError(f"输入图片的形状 {images.shape[1:]} 与模型期望的形状 {self.input_shape} 不匹配！")

    def train(self, epochs):
        best_acc = 0.0
        best_recall = 0.0
        model_save_path = f"model_{self.timestamp}.pth"
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                print("input shape:", images.shape)  # 打印输入图片shape
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self._check_input_shape(images)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                except Exception as e:
                    print(f"训练过程中发生错误: {e}")
            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # 每个epoch结束后在测试集上评估准确率和召回率
            acc, recall = self.test()

            # Store metrics
            self.train_losses.append(avg_loss)
            self.val_accuracies.append(acc)
            self.val_recalls.append(recall)
            self.epoch_dates.append(datetime.datetime.now().strftime("%Y-%m-%d"))  # 新增：记录日期

            if acc > best_acc or recall > best_recall:
                best_acc = max(acc, best_acc)
                best_recall = max(recall, best_recall)
                print(f"New best (acc: {best_acc:.4f}, recall: {best_recall:.4f}), saving model to {model_save_path}...")
                torch.save(self.model.state_dict(), model_save_path)

    def test(self, test_loader=None):
        """
        测试模型，支持自定义测试集。
        :param test_loader: 可选，传入自定义的DataLoader作为测试集。
        """
        self.model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
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
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                except Exception as e:
                    print(f"测试过程中发生错误: {e}")

        accuracy = correct / total if total > 0 else 0
        recall = recall_score(all_labels, all_preds, average='macro')
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Recall (macro): {recall:.4f}")
        return accuracy, recall

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

    def plot_metrics(self):
        """
        绘制训练过程中的损失、准确率和召回率曲线并保存。
        """
        save_path = f"training_metrics_{self.timestamp}.png"
        # 横坐标改为日期
        x_labels = self.epoch_dates if hasattr(self, 'epoch_dates') and len(self.epoch_dates) == len(self.train_losses) else list(range(1, len(self.train_losses) + 1))

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(x_labels, self.train_losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Date')
        plt.ylabel('Loss')
        plt.xticks(rotation=45)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(x_labels, self.val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Date')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(x_labels, self.val_recalls, label='Validation Recall')
        plt.title('Validation Recall')
        plt.xlabel('Date')
        plt.ylabel('Recall')
        plt.xticks(rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training metrics plot saved to {save_path}")
        plt.close() # Close the plot to free memory