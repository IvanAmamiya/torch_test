import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # Adjust labels once here to avoid doing it in each call to __getitem__
        self.img_labels.iloc[:, 1] = self.img_labels.iloc[:, 1].apply(lambda x: x - 1 if x > 0 else x)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # Fixed img_path
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 数据集加载
dataset = CustomImageDataset(
    annotations_file='dataset/annotations.csv',
    img_dir='dataset/images',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型定义
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 128 * 3, 128),
    nn.ReLU(),
    nn.Linear(128, 3)  # 输出3个类别
)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练循环
for epoch in range(3000):
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 模型评估模式
model.eval()

def show_image(img, label):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'Label: {label}')
    plt.show()

# 预测单张图片
def predict_image(image_path, model, transform, class_names=None):
    """
    预测单张图片的类别
    参数：
        image_path: 图片路径
        model: 训练好的模型
        transform: 数据预处理流程
        class_names: 可选的类别名称列表
    """
    # 加载图片
    image = Image.open(image_path).convert("RGB")
    
    # 应用预处理
    transformed_image = transform(image)
    
    # 添加batch维度 (1, C, H, W)
    batch_image = transformed_image.unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        output = model(batch_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = torch.max(probabilities).item()
    
    # 可视化
    plt.imshow(image)
    print(predicted_class)
    title = f'Predicted: {predicted_class}'
    if class_names:
        title += f' ({class_names[predicted_class]})'
    title += f'\nConfidence: {confidence:.2f}'
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    return predicted_class, confidence

# 使用示例（需要根据实际情况调整）：
class_names = ['cat', 'dog', 'donkey']  # 更新类别名称

# 注意要使用与训练相同的transform（这里补充了Normalize会更规范）
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准归一化
                         std=[0.229, 0.224, 0.225])
])

# 进行预测
predict_image(
    image_path='dataset/images/test6.jpg',  # 替换为你的图片路径
    model=model,
    transform=transform,
    class_names=class_names
)
