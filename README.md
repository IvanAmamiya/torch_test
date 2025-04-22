# Torch Test Project

## Project Structure
```
app/
    data_loader.py
    main.py
    trainer.py
    models/
        model.py
    routers/
        __init__.py
        predict.py
    utils/
        __init__.py
data/
    download_data.sh
    FashionMNIST/
        raw/
            train-images-idx3-ubyte
            train-labels-idx1-ubyte
            t10k-images-idx3-ubyte
            t10k-labels-idx1-ubyte
tests/
    test_data_loader.py
    test_main.py
    test_trainer.py
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd torch_test
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download datasets:
   ```bash
   bash data/download_data.sh
   ```

4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Run tests:
   ```bash
   pytest tests/
   ```

## Usage
- Access the API at `http://127.0.0.1:8000`
- Use `/predict` endpoint for model predictions.
- Use `/health` endpoint for health checks.