import pytest
from app.data_loader import load_data

def test_load_data():
    # Add a mock test for load_data function
    data = load_data()
    assert data is not None, "Data should not be None"
    assert isinstance(data, dict), "Data should be a dictionary"
