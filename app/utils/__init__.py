import logging
import matplotlib.pyplot as plt
import os

def setup_logger(name: str, level: int = logging.INFO):
    """Set up a logger with the specified name and level."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def preprocess_data(data):
    """Placeholder for data preprocessing logic."""
    # Add preprocessing steps here
    return data

def plot_training_results():
    epochs = list(range(1, 31))  # Epochs from 1 to 30
    accuracies = [
        0.8559, 0.8440, 0.8502, 0.8583, 0.8608, 0.8762, 0.8720, 0.8784, 0.8809, 0.8777,
        0.8796, 0.8802, 0.8799, 0.8780, 0.8806, 0.8816, 0.8776, 0.8805, 0.8790, 0.8836,
        0.8785, 0.8833, 0.8804, 0.8801, 0.8809, 0.8811, 0.8802, 0.8817, 0.8800, 0.8814
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', label='Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()

    return save_plot_with_unique_name(plt)

def save_plot_with_unique_name(plot, directory="plots"):
    """Save a plot with a unique name in the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    import random
    random_suffix = random.randint(1000, 9999)
    filename = f"training_accuracy_{random_suffix}.png"
    filepath = os.path.join(directory, filename)

    plot.savefig(filepath)
    return filepath