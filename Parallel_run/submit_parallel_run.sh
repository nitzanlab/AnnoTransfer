#!/bin/bash

# Load shared configuration
source ~/.config/annoTransfer.conf || exit 1
: "${PROJECT_DIR:?}" "${VENV_NAME:?}" "${TMP_DIR:?}" "${CACHE_DIR:?}"

# run tasker and dispatcher
job_id=$(sbatch $PROJECT_DIR/Parallel_run/tasker_and_dispatcher.sh | awk '{print $4}') && \
echo "Tasker job submitted with ID: $job_id" && \
tail -F main_controller.out main_controller.err