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

router = APIRouter()

# Load the model (ensure the model is trained and saved beforehand)
model = load_model(weights=None)

# Check if model.pth exists before loading
model_path = "model.pth"
try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print("Model weights partially loaded. Some layers may not match.")
    else:
        print("Warning: model.pth not found. Please ensure the model is trained and saved.")
except RuntimeError as e:
    print("Error loading model weights. The model architecture may have changed. Please retrain the model.")
    print(e)

model.eval()

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
        # Load the pre-trained model weights
        model.load_state_dict(torch.load("model.pth"))
        model.eval()

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

@router.get("/train/")
async def train_model():
    global training_progress
    if training_lock.locked():
        return StreamingResponse(progress_stream(), media_type="text/event-stream")

    async def train_and_stream():
        training_progress = []  # Reset progress
        try:
            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Use CIFAR10Loader for both train and test
            data_loader = CIFAR10Loader(batch_size=64)
            train_loader, test_loader = data_loader.get_loaders()

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Define a learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            # Trainer uses CIFAR-10 train and test loaders
            trainer = Trainer(model, train_loader, test_loader, device, criterion, optimizer)

            # Train the model
            epochs = 50  # Updated to 5 epochs
            for epoch in range(epochs):
                training_progress.append(f"Epoch {epoch + 1}/{epochs} started")
                yield f"data: {training_progress[-1]}\n\n"

                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    # Update progress every 50 batches
                    if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                        batch_progress = f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                        training_progress.append(batch_progress)
                        yield f"data: {batch_progress}\n\n"

                # Step the learning rate scheduler
                scheduler.step()

                # Calculate accuracy for the current epoch
                epoch_accuracy = trainer.calculate_accuracy()
                training_progress.append(f"Epoch {epoch + 1}/{epochs} Accuracy: {epoch_accuracy:.4f}")
                yield f"data: Epoch {epoch + 1}/{epochs} Accuracy: {epoch_accuracy:.4f}\n\n"

                training_progress.append(f"Epoch {epoch + 1}/{epochs} completed")
                yield f"data: {training_progress[-1]}\n\n"

            # Save the trained model
            torch.save(model.state_dict(), "model.pth")
            training_progress.append("Model saved to model.pth.")
            yield f"data: Model saved to model.pth.\n\n"

            # Calculate and display accuracy after training
            accuracy = trainer.calculate_accuracy()
            training_progress.append(f"Final Test Accuracy: {accuracy:.4f}")
            yield f"data: Final Test Accuracy: {accuracy:.4f}\n\n"

            training_progress.append("Training completed successfully.")
            yield f"data: Training completed successfully.\n\n"
        except Exception as e:
            error_message = f"Error: {str(e)}"
            training_progress.append(error_message)
            yield f"data: {error_message}\n\n"

    return StreamingResponse(train_and_stream(), media_type="text/event-stream")