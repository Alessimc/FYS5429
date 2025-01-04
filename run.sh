#!/bin/bash

# Function to set up the environment
setup_environment() {
  if command -v conda &> /dev/null; then
    echo "Conda detected. Setting up the environment with Conda..."
    conda env create -f environment/environment.yml || echo "Environment already exists. Skipping creation."
    conda activate ml-project || source activate ml-project
  else
    echo "Conda not found. Falling back to pip..."
    pip install -r environment/requirements.txt
  fi
}

# Run the environment setup
setup_environment

# Run the main script
echo "Running the main script..."
echo "main does not exist yet. Only environment is created."
# python src/main.py
