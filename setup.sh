#!/usr/bin/env bash
set -e  # Stop on any error

# -------- Configuration --------
# Virtual environment path (change if you want it on an external drive)
VENV_PATH="./object_tracker"

# -------- Check Python version --------
PYTHON=$(which python3)
PYTHON_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [[ "$PYTHON_VER" < "3.10" ]]; then
    echo "Python 3.10+ is required. Current version: $PYTHON_VER"
    exit 1
fi

# -------- Create virtual environment --------
echo "Creating virtual environment at $VENV_PATH..."
$PYTHON -m venv $VENV_PATH

# -------- Activate environment --------
echo "Activating virtual environment..."
source $VENV_PATH/bin/activate

# -------- Upgrade pip --------
echo "Upgrading pip..."
pip install --upgrade pip

# -------- Install PyTorch CPU-only --------
echo "Installing PyTorch CPU-only..."
pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# -------- Install remaining packages --------
echo "Installing remaining packages..."
pip install --no-cache-dir addict==2.4.0 yapf==0.43.0 pycocotools==2.0.10 timm==1.0.19 opencv-python==4.12.0.88 supervision==0.26.1 transformers==4.56.1
# Install Python venv package (Ubuntu/Debian)

sudo apt install python3.10-venv wget

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install opencv-contrib-python==4.7.0.72

# Navigate to GroundingDINO folder
cd GroundingDINO

# Create weights directory and download model
mkdir -p weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Return to GroundingDINO root
cd ..

# -------- Done --------
echo "Setup complete!"
echo "To activate your environment later, run:"
echo "  source $VENV_PATH/bin/activate"
echo "To run your code, use:"
echo "  python3 src/object_tracker_cli.py <path_to_data_file> --prompt \"input prompt\" --output <path_to_data_output>"
