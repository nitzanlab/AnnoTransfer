#!/bin/bash
set -e

# Load shared configuration
source ~/.config/annoTransfer.conf || exit 1
: "${PROJECT_DIR:?}" "${VENV_NAME:?}" "${TMP_DIR:?}" "${CACHE_DIR:?}"

# Set the name of the dataset you want to run
DATASET_NAME="pbmc_healthy"
### USER CONFIGURATION END ###

# SLURM Configuration (unchanged)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,vmem:32G
#SBATCH --cpus-per-task=16
#SBATCH --mem=42G
#SBATCH --time=23:00:00
#SBATCH --output=slurm_out/out.%j
#SBATCH --error=slurm_out/err.%j

# Create output directory
mkdir -p "${WORKDIR}/slurm_out"

# Validate critical paths
if [ ! -f "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please create virtual environment or update VENV_PATH in the script"
    exit 1
fi

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found at $PROJECT_DIR"
    echo "Please clone the project or update PROJECT_DIR in the script"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH"
echo "Activated virtual environment: $(which python)"
python --version

# Start Python script
python3 -u "${PROJECT_DIR}/optimal_compositions.py" ${DATASET_NAME}