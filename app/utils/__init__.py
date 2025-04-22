import logging

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