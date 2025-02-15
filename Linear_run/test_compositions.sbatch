#!/bin/bash

#SBATCH --job-name=test_compositions
#SBATCH --time=3-00:00:00
#SBATCH --output=test_compositions_%j.log
#SBATCH --error=test_compositions_%j.err
#SBATCH --gres=gpu:1,vmem:32G
#SBATCH --cpus-per-task=16
#SBATCH --mem=42G

# Load shared configuration
source ~/.config/annoTransfer.conf || exit 1
: "${PROJECT_DIR:?}" "${VENV_NAME:?}" "${VENV_PATH:?}" "${TMP_DIR:?}" "${CACHE_DIR:?}"
: "${DATASET_NAME:?}" "${TRANS_DATASET_NAME:?}"

# Validate critical paths
if [ ! -f "$VENV_PATH/bin/activate" ]; then
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
source "$VENV_PATH/bin/activate"
echo "Activated virtual environment: $(which python)"
python --version

# define for python where to look for modules
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PROJECT_DIR

# Start Python script
python3 -u "${PROJECT_DIR}/Linear_run/optimal_compositions.py" ${DATASET_NAME} ${TRANS_DATASET_NAME}
