#!/bin/bash

# Load shared configuration
source ~/.config/annoTransfer.conf || exit 1
: "${PROJECT_DIR:?}" "${VENV_NAME:?}" "${TMP_DIR:?}" "${CACHE_DIR:?}"

# run tasker and dispatcher
job_id=$(sbatch $PROJECT_DIR/Parallel_run/tasker_and_dispatcher.sh | awk '{print $4}') && \
echo "tasker_and_dispatcher job submitted with ID: $job_id"

# Wait for SLURM files to be created
while [ ! -f "main_controller_${job_id}.out" ] || [ ! -f "main_controller_${job_id}.err" ]; do
    sleep 1
done

# Start tailing after files exist
tail -F main_controller_$job_id.out main_controller_$job_id.err