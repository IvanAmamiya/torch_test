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

class CIFAR10Loader:
    def __init__(self, batch_size):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # RandAugment(), # 使用 RandAugment 替代 RandomErasing (已移除)
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0), # 去掉随机擦除
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 原归一化
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # 使用 CIFAR-10 常用均值和标准差
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_loaders(self):
        return MixupDataLoader(self.train_loader), self.test_loader

class MixupDataLoader:
    def __init__(self, dataloader, alpha=0.4):
        self.dataloader = dataloader
        self.alpha = alpha

    def __iter__(self):
        for x, y in self.dataloader:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            # One-hot编码标签
            y_onehot = torch.zeros(batch_size, 10, device=y.device)
            y_onehot.scatter_(1, y.view(-1, 1), 1)
            mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index, :]
            yield mixed_x, mixed_y

    def __len__(self):
        return len(self.dataloader)

class CutMixDataLoader:
    def __init__(self, dataloader, alpha=0.4):
        self.dataloader = dataloader
        self.alpha = alpha

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __iter__(self):
        for x, y in self.dataloader:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size)
            y_onehot = torch.zeros(batch_size, 10, device=y.device)
            y_onehot.scatter_(1, y.view(-1, 1), 1)
            y_shuffled = y_onehot[index, :]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
            mixed_y = lam * y_onehot + (1 - lam) * y_shuffled
            yield x, mixed_y

import random
class RandomAugmentDataLoader:
    def __init__(self, dataloader, alpha=0.2):
        self.dataloader = dataloader
        self.alpha = alpha
        self.mixup = MixupDataLoader(dataloader, alpha)
        self.cutmix = CutMixDataLoader(dataloader, alpha)

    def __iter__(self):
        mixup_iter = iter(self.mixup)
        cutmix_iter = iter(self.cutmix)
        for x, y in self.dataloader:
            mode = random.choice(['mixup', 'cutmix', 'none'])
            if mode == 'mixup':
                yield next(mixup_iter)
            elif mode == 'cutmix':
                yield next(cutmix_iter)
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