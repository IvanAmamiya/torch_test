# 数据加载模块
# This file has been moved to app/utils/transforms.py

import os
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST
from PIL import Image
import pandas as pd
import numpy as np
import torch
import random

class CIFAR10Loader:
    def __init__(self, batch_size, num_workers=2, mixup_alpha=0.4):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.mixup_alpha = mixup_alpha

    def get_loaders(self):
        return MixupDataLoader(self.train_loader, alpha=self.mixup_alpha), self.test_loader

class MixupDataLoader:
    def __init__(self, dataloader, alpha=0.4):
        self.dataloader = dataloader
        self.alpha = alpha

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

class RandomAugmentDataLoader:
    def __init__(self, dataloader, mixup_alpha=0.4):
        self.dataloader = dataloader
        self.mixup = MixupDataLoader(dataloader, mixup_alpha)

    def __iter__(self):
        mixup_iter = iter(self.mixup)
        for x, y in self.dataloader:
            mode = random.choice(['mixup', 'none'])
            if mode == 'mixup':
                yield next(mixup_iter)
            else:
                # One-hot编码标签
                y_onehot = torch.zeros(x.size(0), 10, device=y.device)
                y_onehot.scatter_(1, y.view(-1, 1), 1)
                yield x, y_onehot

    def __len__(self):
        return len(self.dataloader)

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
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        # 修正：PIL Image 没有 shape 属性，如需 shape 用 tensor/array
        # print(f"[CustomImageDataset] idx={idx}, image shape={image.shape}, dtype={image.dtype}, min={image.min().item()}, max={image.max().item()}, label={label}")
        if idx < 10:
            if hasattr(image, 'shape'):
                print(f"[CustomImageDataset] idx={idx}, image shape={image.shape}, dtype={image.dtype}, min={image.min().item()}, max={image.max().item()}, label={label}")
            else:
                print(f"[CustomImageDataset] idx={idx}, image size={getattr(image, 'size', None)}, label={label}")
        return image, label

class DatasetLoader:
    def __init__(self, annotations_file, img_dir, transform):
        self.dataset = CustomImageDataset(annotations_file, img_dir, transform)

    def get_loaders(self, batch_size, shuffle=True):
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader