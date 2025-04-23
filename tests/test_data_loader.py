import pytest
from app.data_loader import FashionMNISTLoader

def test_fashion_mnist_loader():
    loader = FashionMNISTLoader(batch_size=32)
    train_loader, test_loader = loader.get_loaders()

    assert train_loader is not None, "Train loader should not be None"
    assert test_loader is not None, "Test loader should not be None"
    assert len(train_loader.dataset) > 0, "Train dataset should not be empty"
    assert len(test_loader.dataset) > 0, "Test dataset should not be empty"
