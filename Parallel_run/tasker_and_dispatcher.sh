#!/bin/bash

# Load shared configuration
source ~/.config/annoTransfer.conf || exit 1
: "${PROJECT_DIR:?}" "${VENV_NAME:?}" "${TMP_DIR:?}" "${CACHE_DIR:?}"

# Set script locations
TASKER_SCRIPT="$PROJECT_DIR/Parallel_run/tasker.py"
WORKER_SCRIPT="$PROJECT_DIR/Parallel_run/worker_script.py"

# ----------------------
# User Configuration
# ----------------------

CSV_FILE="pbmc_healthy_worker_jobs.csv"
RESULTS_DIR="results"
CHUNK_SIZE=200 # Number of jobs script will submit at once
MAX_JOBS_IN_QUEUE=1000 # script will wait if this many jobs are already in queue

# ----------------------
# Path Validation
# ----------------------
validate_path() {
    if [ ! -e "$1" ]; then
        echo "ERROR: Required path not found - $1"
        exit 1
    fi
}

validate_path "$VENV_PATH"
validate_path "$TASKER_SCRIPT"
validate_path "$WORKER_SCRIPT"

# define for python where to look for modules
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PROJECT_DIR

# ----------------------
# Script Logic
# ----------------------

echo "Submitting tasker job"
# Run the tasker as sbatch job
job_id=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="\$PYTHONPATH:$PROJECT_DIR"
source "$VENV_PATH/bin/activate"
python3 -u "$TASKER_SCRIPT" 2>&1
EOF
)

# Wait for the SLURM output file to be created
output_file="slurm-${job_id}.out"
while [ ! -f "$output_file" ]; do
    sleep 1
done

# Show real-time output in terminal
tail -F "$output_file" &
tail_pid=$!

# Cleanup function
cleanup() {
    kill $tail_pid 2>/dev/null
}
trap cleanup EXIT

# Wait for job completion
while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
    sleep 10
done

# 1) Create results directory
mkdir -p "$RESULTS_DIR"

# 2) Count total data rows
TOTAL_ROWS=$(($(wc -l < "$CSV_FILE") - 1))
[ "$TOTAL_ROWS" -le 0 ] && { echo "Error: No data rows in $CSV_FILE."; exit 1; }

# 3) Calculate chunks
NUM_CHUNKS=$(( (TOTAL_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "Dispatching $NUM_CHUNKS chunks total for $TOTAL_ROWS rows..."

# 4) Job queue monitoring
jobs_in_queue() {
    squeue -u "$USER" -h | wc -l
}

# 5) Chunk submission loop
for ((i=0; i<NUM_CHUNKS; i++)); do
    OFFSET=$((i * CHUNK_SIZE))
    END=$((OFFSET + CHUNK_SIZE - 1))
    [ "$END" -ge "$TOTAL_ROWS" ] && END=$((TOTAL_ROWS - 1))

    # Throttle jobs
    while [ "$(jobs_in_queue)" -ge "$MAX_JOBS_IN_QUEUE" ]; do
        echo "Queue full ($(jobs_in_queue)/$MAX_JOBS_IN_QUEUE). Waiting 5 minutes..."
        sleep 300
    done

    echo "Submitting chunk $i (OFFSET=$OFFSET)..."

    # 6) Retry submission logic
    while true; do
        submission_output=$(sbatch --parsable <<EOT 2>&1
#!/bin/bash
#SBATCH --array=0-$((CHUNK_SIZE-1))
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=${RESULTS_DIR}/out.%A_%a
#SBATCH --error=${RESULTS_DIR}/err.%A_%a
#SBATCH --killable
#SBATCH --requeue

source "$VENV_PATH"

ROW_ID=\$((SLURM_ARRAY_TASK_ID + $OFFSET))

if [ "\$ROW_ID" -le "$END" ]; then
    srun python "$WORKER_SCRIPT" \\
        --csv_file="$CSV_FILE" \\
        --row_id="\$ROW_ID" \\
        --device="cuda" \\
        --epoch_num=8 \\
        --batch_size=128 \\
        --model_name="mlp" \\
        --output_dir="results"
else
    echo "Skipping row \$ROW_ID as it exceeds $END."
fi
EOT
        )

        if [[ "$submission_output" =~ ^[0-9]+$ ]]; then
            echo "Chunk $i submitted with job ID $submission_output"
            break
        else
            echo "Submission issue: ${submission_output##*: } - Retrying in 5 minutes..."
            sleep 300
        fi
    done
done

echo "All $NUM_CHUNKS chunks submitted (queue limit: $MAX_JOBS_IN_QUEUE)."
