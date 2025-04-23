import sys
import os
import torch
from torch import nn, optim
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict  # Changed to absolute import
from .data_loader import CIFAR10Loader
from .models.model import load_model
from .trainer import Trainer

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set default batch size
batch_size = 64

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

@app.post("/train")
def train_model():
    # Use the default batch size
    data_loader = CIFAR10Loader(batch_size)
    train_loader, test_loader = data_loader.get_loaders()

    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    trainer = Trainer(model, train_loader, test_loader, device, criterion, optimizer)
    trainer.train(epochs=5)

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    return {"message": "Model training completed and saved as model.pth"}

if __name__ == "__main__":
    import os
    print("Run the application using the following command:")
    print(f"uvicorn {os.path.basename(__file__).replace('.py', '')}:app --host 127.0.0.1 --port 8000 --reload")