#!/bin/bash

# exit immediately on non-zero status
set -e

. "$(conda info --base)/etc/profile.d/conda.sh"

# create a new conda environment
conda create -y --force -n wri_cmnist python=3.9.15

# activate it
conda activate wri_cmnist

# install requirements
yes | pip install --upgrade pip
yes | pip install -r requirements.txt
