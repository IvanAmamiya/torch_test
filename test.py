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


# 定义模型
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

# 初始化模型和优化器
model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
scaler = GradScaler()

# 训练模型
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = (labels - 1).to(device)  # ⚠️ CrossEntropyLoss 需要标签从 0 开始

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted += 1  # ⚠️ 模型预测为 0~4，对应标签 1~5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f'Validation Accuracy after Epoch {epoch+1}: {acc:.4f}')

# 预测函数
def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path).convert("L")
    transformed_image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(transformed_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item() + 1
        confidence = torch.max(probabilities).item()

    plt.imshow(np.array(image), cmap="gray")
    plt.title(f'Predicted: {predicted_class} ({class_names[predicted_class-1]})\nConfidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()

    return predicted_class, confidence

# 类别名（显示用）
class_names = ['cat', 'dog', 'donkey', 'horse', 'zebra']

# 示例预测
predict_image('./dataset/images/test4.jpg', model, transform, class_names)