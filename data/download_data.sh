#!/bin/bash

# Download CIFAR-10 dataset
if [ ! -f "data/cifar-10-python.tar.gz" ]; then
  echo "Downloading CIFAR-10 dataset..."
  wget -P data/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
else
  echo "CIFAR-10 dataset already exists."
fi

# Download FashionMNIST dataset
if [ ! -d "data/FashionMNIST/raw" ]; then
  echo "Downloading FashionMNIST dataset..."
  mkdir -p data/FashionMNIST/raw
  wget -P data/FashionMNIST/raw/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
  wget -P data/FashionMNIST/raw/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
  wget -P data/FashionMNIST/raw/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
  wget -P data/FashionMNIST/raw/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
else
  echo "FashionMNIST dataset already exists."
fi