#!/bin/bash

# Exit on error
set -e

echo "Setting up Jupyter environment for agentic data analysis..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install from https://www.python.org/downloads/"
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is required but not installed. Please install pip first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is required but not installed. Please install from https://nodejs.org/"
    exit 1
fi

# Install JupyterLab
echo "Installing JupyterLab..."
pip install jupyterlab

# Install tslab globally
echo "Installing tslab globally..."
npm install -g tslab

# Install tslab kernel
echo "Installing tslab kernel..."
tslab install --version
tslab install --python=python3.12

# Verify kernel installation
echo "Verifying kernel installation..."
jupyter kernelspec list

# Build and link the package
echo "Building and linking the package..."
cd ../langgraph
npm run build
npm run link:local

echo "Setup complete! You can now run 'jupyter lab' to start JupyterLab." 