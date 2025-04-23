import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv
from torch.cuda.amp import autocast, GradScaler
from app.data_loader import DatasetLoader

# 自动使用 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("L")
        label = int(self.img_labels.iloc[idx, 1])  # 显式转 int 防止类型异常
        if self.transform:
            image = self.transform(image)
        return image, label  # ✅ 正确返回 image 和 int label

# 图像增强与预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Update paths to valid CSV and image directory for testing
if not os.path.exists('./dataset/annotations.csv') or not os.path.exists('./dataset/images'):
    print("Dataset not found. Using default dataset.")
    annotations_file = './data/FashionMNIST/annotations.csv'  # Example valid path
    img_dir = './data/FashionMNIST/images'  # Example valid path
    os.makedirs(img_dir, exist_ok=True)  # Ensure the directory exists
    # Create a mock annotations file for testing
    with open(annotations_file, 'w') as f:
        f.write("image,label\nmock_image.png,1\n")
    # Create a mock image file for testing
    from PIL import Image
    mock_image_path = os.path.join(img_dir, 'mock_image.png')
    Image.new('L', (28, 28)).save(mock_image_path)
    dataset_loader = DatasetLoader(annotations_file, img_dir, transform)
else:
    dataset_loader = DatasetLoader('./dataset/annotations.csv', './dataset/images', transform)

# Ensure the test image exists for prediction
mock_test_image_path = './dataset/images/test4.jpg'
os.makedirs(os.path.dirname(mock_test_image_path), exist_ok=True)
Image.new('L', (128, 128)).save(mock_test_image_path)

# 模型定义模块
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 128, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 个类别

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 模型训练模块
class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, device, criterion, optimizer, scaler):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = (labels - 1).to(self.device)  # ⚠️ CrossEntropyLoss 需要标签从 0 开始

                self.optimizer.zero_grad()
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            self.validate(epoch)

    def validate(self, epoch):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predicted += 1  # ⚠️ 模型预测为 0~4，对应标签 1~5
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        print(f'Validation Accuracy after Epoch {epoch+1}: {acc:.4f}')

# 数据集加载模块
class DatasetLoader:
    def __init__(self, annotations_file, img_dir, transform):
        self.dataset = CustomImageDataset(annotations_file, img_dir, transform)

    def get_loaders(self, batch_size, shuffle=True):
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

# 图像预测模块
class ImagePredictor:
    def __init__(self, model, transform, class_names, device):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.device = device

    def predict(self, image_path):
        image = Image.open(image_path).convert("L")
        transformed_image = self.transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(transformed_image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item() + 1
            confidence = torch.max(probabilities).item()

        plt.imshow(np.array(image), cmap="gray")
        plt.title(f'Predicted: {predicted_class} ({self.class_names[predicted_class-1]})\nConfidence: {confidence:.2f}')
        plt.axis('off')
        plt.show()

        return predicted_class, confidence

# 初始化数据集加载器
train_loader, test_loader = dataset_loader.get_loaders(batch_size=32)

# 初始化模型和优化器
model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
scaler = GradScaler()

# 示例训练和预测
trainer = ModelTrainer(model, train_loader, test_loader, device, criterion, optimizer, scaler)
trainer.train(epochs=10)

predictor = ImagePredictor(model, transform, ['cat', 'dog', 'donkey', 'horse', 'zebra'], device)
predictor.predict('./dataset/images/test4.jpg')