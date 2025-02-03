#!/bin/bash
set -e  # Exit on error

# ----------------------
# User Configuration
# ----------------------
# 1. Set your project root directory
PROJECT_DIR="$HOME/annotatability"  # Example: "$HOME/my_project"

# 2. Virtual environment name and paths
VENV_NAME="annot_venv"
VENV_PATH="$PROJECT_DIR/$VENV_NAME"

# 3. Temporary directories configuration
TMPDIR_PATH="$PROJECT_DIR/tmp"        # Custom TMPDIR
CACHE_DIR_PATH="$PROJECT_DIR/cache"   # Custom pip cache

# 4. Requirements file location
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

# ----------------------
# Directory Setup
# ----------------------
# Create temporary directories if they don't exist
mkdir -p "$TMPDIR_PATH"
mkdir -p "$CACHE_DIR_PATH"

# ----------------------
# Setup Execution
# ----------------------
echo "=== Setting up virtual environment: $VENV_NAME ==="

# Step 1: Create virtual environment
echo "Creating virtual environment in: $VENV_PATH"
python3 -m venv "$VENV_PATH"

# Step 2: Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Step 3: Configure environment variables
echo "Configuring temporary directories..."
export TMPDIR="$TMPDIR_PATH"
export PIP_CACHE_DIR="$CACHE_DIR_PATH"

# Step 4: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 5: Install PyTorch
echo "Installing PyTorch..."
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117

# Step 6: Install other requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing additional packages from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Warning: Requirements file not found at $REQUIREMENTS_FILE"
    echo "Skipping additional package installation."
fi

# Step 7: Verify installation
echo "Verifying PyTorch installation..."
python -c "import torch; print('Torch version:', torch.__version__)"

# Completion message
echo "=== Virtual environment $VENV_NAME setup complete! ==="
echo "To activate the environment, run:"
echo "    source $VENV_PATH/bin/activate"