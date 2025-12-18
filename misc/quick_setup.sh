#!/bin/bash
# Quick setup script for MineStudio + STEVE-1 + AgentBeats evaluation
# This script automates the environment setup process

set -e  # Exit on error

echo "=========================================="
echo "MineStudio + STEVE-1 Quick Setup"
echo "=========================================="
echo ""

# Detect system
echo "Detecting system..."
OS=$(uname -s)
ARCH=$(uname -m)
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')

echo "  OS: $OS"
echo "  Architecture: $ARCH"
echo "  Python: $PYTHON_VERSION"
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    USE_CONDA=true
    echo "✓ Conda detected - will use conda environment"
else
    USE_CONDA=false
    echo "⚠ Conda not found - will use venv instead"
    echo "  (Install conda from: https://docs.conda.io/en/latest/miniconda.html)"
fi

# Create environment
ENV_NAME="minestudio"

if [ "$USE_CONDA" = true ]; then
    echo ""
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y || {
        echo "⚠ Environment already exists, activating..."
    }
    
    echo "Activating environment..."
    # Initialize conda for bash
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    
    # Verify we're using the right Python
    echo "Verifying Python version in environment..."
    python --version
    which python
    which pip
    
    echo "Installing OpenJDK 8..."
    conda install --channel=conda-forge openjdk=8 -y
    
    echo "Verifying Java installation..."
    java -version
    
else
    echo ""
    echo "Creating venv environment: ${ENV_NAME}_env"
    python3.10 -m venv ${ENV_NAME}_env || python3 -m venv ${ENV_NAME}_env
    
    echo "Activating environment..."
    source ${ENV_NAME}_env/bin/activate
    
    echo "⚠ Please install OpenJDK 8 manually:"
    if [ "$OS" = "Darwin" ]; then
        echo "  brew install --cask temurin8"
    elif [ "$OS" = "Linux" ]; then
        echo "  sudo apt-get install openjdk-8-jdk"
    fi
fi

# Upgrade pip (use python -m pip to ensure we use the right Python)
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install core dependencies first
echo ""
echo "Installing core dependencies..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pillow opencv-python huggingface-hub rich tqdm einops psutil

# Install MineStudio (this will pull in its dependencies)
echo ""
echo "Installing MineStudio..."
python -m pip install minestudio

# Set environment variable for auto-download
export AUTO_DOWNLOAD_ENGINE=1

# Verify installation
echo ""
echo "Verifying installation..."
python3 verify_minestudio.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment:"
if [ "$USE_CONDA" = true ]; then
    echo "     conda activate $ENV_NAME"
else
    echo "     source ${ENV_NAME}_env/bin/activate"
fi
echo ""
echo "  2. Test STEVE-1 inference:"
echo "     python3 run_steve1_inference.py"
echo ""
echo "  3. Run evaluation:"
echo "     python3 run_agentbeats_evaluation.py --task collect_wood --difficulty simple"
echo ""

