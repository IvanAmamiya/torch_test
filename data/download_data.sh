#!/bin/bash

# Check if FashionMNIST dataset is already extracted
if [ -f "data/FashionMNIST/raw/train-images-idx3-ubyte" ]; then
  echo "FashionMNIST dataset already extracted. Skipping download."
else
  echo "Downloading and extracting FashionMNIST dataset..."
  mkdir -p data/FashionMNIST/raw
  wget -P data/FashionMNIST/raw/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
  wget -P data/FashionMNIST/raw/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
  wget -P data/FashionMNIST/raw/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
  wget -P data/FashionMNIST/raw/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

  # Extract files
  gunzip data/FashionMNIST/raw/*.gz
fi

# Cleanup temporary files
if [ -f "data/FashionMNIST/raw/*.gz" ]; then
  echo "Cleaning up temporary files..."
  rm data/FashionMNIST/raw/*.gz
fi