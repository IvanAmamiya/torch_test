from fastapi import APIRouter, UploadFile
import torch
from torchvision import transforms
from PIL import Image
from ..models.model import load_model, predict
from ..trainer import Trainer
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
import threading
import os
import time
from fastapi.responses import StreamingResponse, HTMLResponse
from typing import Generator
import asyncio
from ..data_loader import CIFAR10Loader
import csv

router = APIRouter()

BATCH_SIZE = 32  # Adjusted batch size for training
# Adjusted batch size for training
TEST_BATCH_SIZE = 64
AUG_BATCH_SIZE = 2048

# 学习率与batch size自适应设置
BASE_LR = 0.1
BASE_BATCH_SIZE = 64
lr = BASE_LR * (BATCH_SIZE / BASE_BATCH_SIZE)

# Load the model (ensure the model is trained and saved beforehand)
# model = load_model(weights=None)

# Check if model.pth exists before loading
# model_path = "model.pth"
# try:
#     if os.path.exists(model_path):
#         model.load_state_dict(torch.load("model.pth"))
#     else:
#         print("Warning: model.pth not found. Please ensure the model is trained and saved.")
# except RuntimeError as e:
#     print("Error loading model weights. The model architecture may have changed. Please retrain the model.")
#     print(e)

# model.eval()

# Training lock to prevent concurrent training
training_lock = threading.Lock()

# Global variable to store training progress
training_progress = []

def progress_stream() -> Generator[str, None, None]:
    while True:
        if training_progress:
            # Log the latest progress to the terminal for debugging
            print(f"SSE Update: {training_progress[-1]}")
            yield f"data: {training_progress[-1]}\n\n"
            time.sleep(1)  # Wait for 1 second before sending the next update
        else:
            time.sleep(0.1)  # Check more frequently when no updates are available

@router.post("/predict/")
async def predict_image(file: UploadFile):
    try:
        # Read the uploaded image
        image = Image.open(file.file).convert("RGB")

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),  # 修改为32x32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        # Ensure the input tensor is on the same device as the model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Perform prediction
        prediction_index = predict(model, input_tensor)

        # Map prediction index to class label
        class_labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        prediction_label = class_labels[prediction_index]

        # Convert image to base64 for display
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return HTML response
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Result</title>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <p><strong>Predicted Class:</strong> {prediction_label}</p>
            <img src="data:image/png;base64,{img_str}" alt="Uploaded Image" style="max-width: 300px;">
            <br><br>
            <a href="/predict/">Upload Another Image</a>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return {"error": str(e)}

@router.get("/predict/")
async def get_predict_form():
    return HTMLResponse(content=open("app/templates/predict_form.html").read(), status_code=200)

def run_training_with_loader(train_loader, test_loader, epochs, device, optimizer_fn, training_progress=None, stream=False):
    model = load_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    trainer = Trainer(model, train_loader, test_loader, device, criterion, optimizer)
    acc_list = []
    for epoch in range(epochs):
        if training_progress is not None:
            training_progress.append(f"Epoch {epoch + 1}/{epochs} started")
            if stream:
                yield f"data: {training_progress[-1]}\n\n"
        trainer.model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if training_progress is not None and ((batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader)):
                batch_progress = f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                training_progress.append(batch_progress)
                if stream:
                    yield f"data: {batch_progress}\n\n"
        scheduler.step()
        epoch_accuracy = trainer.calculate_accuracy()
        acc_list.append(epoch_accuracy)
        if training_progress is not None:
            training_progress.append(f"Epoch {epoch + 1}/{epochs} Accuracy: {epoch_accuracy:.4f}")
            if stream:
                yield f"data: Epoch {epoch + 1}/{epochs} Accuracy: {epoch_accuracy:.4f}\n\n"
            training_progress.append(f"Epoch {epoch + 1}/{epochs} completed")
            if stream:
                yield f"data: {training_progress[-1]}\n\n"
    # trainer.plot_metrics()  # 可选：如需每次都保存曲线
    if stream:
        return
    return acc_list[-1] if acc_list else 0.0

@router.get("/train/")
async def train_model():
    global training_progress
    if training_lock.locked():
        return StreamingResponse(progress_stream(), media_type="text/event-stream")

    async def train_and_stream():
        training_progress = []  # Reset progress
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data_loader = CIFAR10Loader(batch_size=BATCH_SIZE)
            train_loader, test_loader = data_loader.get_loaders()
            optimizer_fn = lambda params: optim.Adam(params, lr=lr)
            gen = run_training_with_loader(train_loader, test_loader, epochs=50, device=device, optimizer_fn=optimizer_fn, training_progress=training_progress, stream=True)
            for msg in gen:
                yield msg
        except Exception as e:
            error_message = f"Error: {str(e)}"
            training_progress.append(error_message)
            yield f"data: {error_message}\n\n"
    return StreamingResponse(train_and_stream(), media_type="text/event-stream")

@router.get("/train_aug_curve/")
async def train_aug_curve():
    import matplotlib.pyplot as plt
    from app.data_loader import MixupDataLoader, CIFAR10Loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_list = [0.0] + [round(x * 0.2, 1) for x in range(1, 6)]  # 0.0(无mixup) + 0.2~1.0，步长0.2
    acc_list = []
    optimizer_fn = lambda params: optim.Adam(params, lr=lr)
    for alpha in alpha_list:
        # 重新初始化模型和数据
        model = load_model().to(device)
        data_loader = CIFAR10Loader(batch_size=BATCH_SIZE)
        train_loader, test_loader = data_loader.get_loaders()
        mixup_loader = MixupDataLoader(train_loader, alpha=alpha)
        acc = run_training_with_loader(mixup_loader, test_loader, epochs=3, device=device, optimizer_fn=optimizer_fn)
        if hasattr(acc, '__iter__') and not isinstance(acc, str):
            acc = list(acc)[-1] if acc else 0.0
        acc_list.append(acc)
    # 绘制折线图
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_list, acc_list, marker='o', label='Test Accuracy')
    plt.title('Mixup Alpha vs Test Accuracy')
    plt.xlabel('Mixup Alpha')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    import random
    random_suffix = random.randint(1000, 9999)
    save_path = os.path.join(plots_dir, f"mixup_alpha_curve_{random_suffix}.png")
    plt.savefig(save_path)
    plt.close()
    return {"message": "Mixup alpha curve experiment completed.", "plot": save_path, "alphas": alpha_list, "accuracies": acc_list}

def save_experiment_csv(filename, alpha_list, all_acc_lists, all_loss_lists, epochs, param_dict=None):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入参数信息
        if param_dict:
            for k, v in param_dict.items():
                writer.writerow([f"#{k}={v}"])
        # 记录所有增强参数
        writer.writerow([f"# mixup_alpha={param_dict.get('mixup_alpha', '')}"])
        writer.writerow([f"# cutmix_alpha={param_dict.get('cutmix_alpha', '')}"])
        writer.writerow([f"# randaugment_n={param_dict.get('randaugment_n', '')}"])
        writer.writerow([f"# randaugment_m={param_dict.get('randaugment_m', '')}"])
        writer.writerow([f"# randomerasing_p={param_dict.get('randomerasing_p', '')}"])
        # 写入表头
        header = ['epoch'] + [f'alpha={a}_acc' for a in alpha_list] + [f'alpha={a}_loss' for a in alpha_list]
        writer.writerow(header)
        for epoch in range(epochs):
            row = [epoch+1]
            for acc_list in all_acc_lists:
                row.append(acc_list[epoch] if epoch < len(acc_list) else '')
            for loss_list in all_loss_lists:
                row.append(loss_list[epoch] if epoch < len(loss_list) else '')
            writer.writerow(row)

@router.get("/train_aug_curve_stream/")
async def train_aug_curve_stream():
    import matplotlib.pyplot as plt
    from app.data_loader import MixupDataLoader, CIFAR10Loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_list = [0.0] + [round(x * 0.1, 1) for x in range(1, 11)]  # 0.0(无mixup) + 0.1~1.0
    all_acc_lists = []
    all_loss_lists = []
    epochs = 30  # 可根据需要调整

    async def stream():
        for alpha in alpha_list:
            optimizer_fn = lambda params: optim.SGD(params, lr=0.01875, momentum=0.9, weight_decay=1e-4, nesterov=True)
            training_progress = []
            data_loader = CIFAR10Loader(batch_size=BATCH_SIZE)
            train_loader, test_loader = data_loader.get_loaders()
            if alpha == 0.0:
                loader = train_loader  # 不用Mixup
            else:
                loader = MixupDataLoader(train_loader, alpha=alpha)
            acc_list = []
            loss_list = []
            gen = run_training_with_loader(loader, test_loader, epochs=epochs, device=device, optimizer_fn=optimizer_fn, training_progress=training_progress, stream=True)
            for msg in gen:
                yield msg
            for item in training_progress:
                if "Accuracy:" in item:
                    try:
                        acc = float(item.split("Accuracy:")[-1])
                        acc_list.append(acc)
                    except Exception:
                        acc_list.append(0.0)
                if "Loss:" in item:
                    try:
                        loss = float(item.split("Loss:")[-1])
                        loss_list.append(loss)
                    except Exception:
                        loss_list.append(0.0)
            all_acc_lists.append(acc_list)
            all_loss_lists.append(loss_list)
            yield f"data: Alpha {alpha} finished.\n\n"
        # 绘制多条精度和loss折线图
        plots_dir = "plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        import random
        random_suffix = random.randint(1000, 9999)
        # 精度曲线
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for i, acc_list in enumerate(all_acc_lists):
            plt.plot(range(1, len(acc_list)+1), acc_list, marker='o', label=f'alpha={alpha_list[i]}')
        plt.title('Mixup Alpha vs Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        acc_save_path = os.path.join(plots_dir, f"mixup_alpha_curve_{random_suffix}.png")
        plt.savefig(acc_save_path)
        plt.close()
        # Loss曲线
        plt.figure(figsize=(10, 6))
        for i, loss_list in enumerate(all_loss_lists):
            plt.plot(range(1, len(loss_list)+1), loss_list, marker='o', label=f'alpha={alpha_list[i]}')
        plt.title('Mixup Alpha vs Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        loss_save_path = os.path.join(plots_dir, f"mixup_alpha_loss_curve_{random_suffix}.png")
        plt.savefig(loss_save_path)
        plt.close()
        # 保存csv，带参数
        param_dict = {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'batch_size': BATCH_SIZE,
            'test_batch_size': TEST_BATCH_SIZE,
            'aug_batch_size': AUG_BATCH_SIZE,
            'base_lr': BASE_LR,
            'actual_lr': lr,
            'epochs': epochs,
            'model': 'CIFAR-ResNet18',
            'optimizer': 'Adam',
            'scheduler': 'StepLR(step_size=5, gamma=0.1)',
            'alpha_list': alpha_list,
            'num_workers': 2,
            'data_augment': 'RandomCrop, RandomHorizontalFlip, ColorJitter, Normalize',
        }
        csv_save_path = os.path.join(plots_dir, f"mixup_alpha_results_{random_suffix}.csv")
        save_experiment_csv(csv_save_path, alpha_list, all_acc_lists, all_loss_lists, epochs, param_dict)
        yield f"data: All alphas finished. Accuracy plot saved to {acc_save_path}, Loss plot saved to {loss_save_path}, CSV saved to {csv_save_path}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

