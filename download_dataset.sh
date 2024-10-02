#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# Download data
HOST="https://github.com/mkolod/MNIST/raw/refs/heads/master/"
wget "$HOST/train-images-idx3-ubyte.gz"
wget "$HOST/train-labels-idx1-ubyte.gz"
wget "$HOST/t10k-images-idx3-ubyte.gz"
wget "$HOST/t10k-labels-idx1-ubyte.gz"

gunzip -d *.gz

