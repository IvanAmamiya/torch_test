import torch
import torch.nn as nn
import torch.optim as optim
from app.data_loader import CIFAR10Loader
from app.models.model import SimpleCNN
from app.trainer import Trainer

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-10 data
    batch_size = 64
    data_loader = CIFAR10Loader(batch_size)
    train_loader, test_loader = data_loader.get_loaders()

    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    trainer = Trainer(model, train_loader, test_loader, device, criterion, optimizer)
    trainer.train(epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    print("Model training completed and saved as model.pth")

if __name__ == "__main__":
    main()