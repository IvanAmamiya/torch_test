import sys
import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict  # Changed to absolute import
from .data_loader import CIFAR10Loader
from .models.model import load_model
from .trainer import Trainer

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set default batch size
batch_size = 1024  # 调整batch_size为1024，适合多卡大显存环境

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI ML Service"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, input, target):
        log_probs = torch.nn.functional.log_softmax(input, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

@app.post("/train")
def train_model():
    # Use the default batch size
    data_loader = CIFAR10Loader(batch_size)
    train_loader, test_loader = data_loader.get_loaders()

    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(pretrained=False).to(device)
    criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # CosineAnnealingLR: T_max为总epoch数
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    # ReduceLROnPlateau: 监控验证集loss，patience=5
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Train the model
    trainer = Trainer(model, train_loader, test_loader, device, criterion, optimizer)
    trainer.scheduler = scheduler
    trainer.train(epochs=50)

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    return {"message": "Model training completed and saved as model.pth"}

if __name__ == "__main__":
    import os
    print("Run the application using the following command:")
    print(f"uvicorn {os.path.basename(__file__).replace('.py', '')}:app --host 127.0.0.1 --port 8000 --reload")