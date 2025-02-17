#!/bin/bash

#SBATCH --job-name=main_controller
#SBATCH --time=3-00:00:00
#SBATCH --output=main_controller_%j.out
#SBATCH --error=main_controller_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Starting main controller script..."
job_id=$SLURM_JOB_ID
echo "output file: main_controller_$job_id.out"

# Load shared configuration
source ~/.config/annoTransfer.conf || exit 1
: "${PROJECT_DIR:?}" "${VENV_NAME:?}" "${TMP_DIR:?}" "${CACHE_DIR:?}"
: "${DATASET_NAME:?}" "${TRANS_DATASET_NAME:?}"

# Set script locations
TASKER_SCRIPT="$PROJECT_DIR/Parallel_run/tasker.py"
WORKER_SCRIPT="$PROJECT_DIR/Parallel_run/worker_script.py"

# ----------------------
# User Configuration
# ----------------------
CSV_FILE="${DATASET_NAME}_worker_jobs.csv"
RESULTS_DIR="results_${DATASET_NAME}_$(date +%Y%m%d_%H%M)"
CHUNK_SIZE=200 # Number of jobs script will submit at once
MAX_JOBS_IN_QUEUE=1000 # script will wait if this many jobs are already in queue

# ----------------------
# Path Validation
# ----------------------
validate_path() {
    [ ! -e "$1" ] && { echo "ERROR: Path not found - $1"; exit 1; }
}

validate_path "$VENV_PATH"
validate_path "$TASKER_SCRIPT"
validate_path "$WORKER_SCRIPT"

# Environment setup
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PROJECT_DIR

# ----------------------
# Script Logic
# ----------------------
echo "Submitting tasker job for dataset: $DATASET_NAME..."
tasker_job_id=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=tasker_${DATASET_NAME}
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=tasker_${DATASET_NAME}_%j.out
#SBATCH --error=tasker_${DATASET_NAME}_%j.err
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
source "$VENV_PATH/bin/activate"
python3 -u "$TASKER_SCRIPT" --dataset "$DATASET_NAME"
EOF
)

echo "Tasker job submitted with ID: $tasker_job_id"
echo "Monitoring tasker output in tasker_${DATASET_NAME}_${tasker_job_id}.out..."

# Wait for SLURM files to be created
while [ ! -f "tasker_${DATASET_NAME}_${tasker_job_id}.out" ] || [ ! -f "tasker_${DATASET_NAME}_${tasker_job_id}.err" ]; do
    sleep 1
done

# Show real-time output in terminal (both .out and .err)
tail -F "tasker_${DATASET_NAME}_${tasker_job_id}.out" "tasker_${DATASET_NAME}_${tasker_job_id}.err" &
tail_pid=$!

cleanup() {
    kill $tail_pid 2>/dev/null
}
trap cleanup EXIT

# Wait for job completion
while squeue -j "$tasker_job_id" 2>/dev/null | grep -q "$tasker_job_id"; do
    sleep 10
done

# Check job status
if ! sacct -j "$tasker_job_id" --format=State | grep -q "COMPLETED"; then
    echo "ERROR: Tasker job $tasker_job_id failed. Check tasker_${tasker_job_id}.err"
    exit 1
fi

# 1) Create results directory
# Remove existing results directory if it exists and create fresh structure
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR/analysis"

# 2) Count total data rows
TOTAL_ROWS=$(($(wc -l < "$CSV_FILE") - 1))
[ "$TOTAL_ROWS" -le 0 ] && { echo "Error: No data rows in $CSV_FILE."; exit 1; }

# 3) Calculate chunks
NUM_CHUNKS=$(( (TOTAL_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "Dispatching $NUM_CHUNKS chunks total for $TOTAL_ROWS rows..."
echo "creating logs in $RESULTS_DIR/logs"

# 4) Job queue monitoring
jobs_in_queue() {
    squeue -u "$USER" -h | wc -l
}

# 5) Chunk submission loop
declare -a chunk_job_ids  # Track all chunk job IDs

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
#SBATCH --job-name=worker_${DATASET_NAME}_chunk_${i}
#SBATCH --array=0-$((CHUNK_SIZE-1))
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=${RESULTS_DIR}/logs/out.%A_%a
#SBATCH --error=${RESULTS_DIR}/logs/err.%A_%a
#SBATCH --killable
#SBATCH --requeue

source "$VENV_PATH/bin/activate"

ROW_ID=\$((SLURM_ARRAY_TASK_ID + $OFFSET))

if [ "\$ROW_ID" -le "$END" ]; then
    srun python "$WORKER_SCRIPT" \\
        --csv_file="$CSV_FILE" \\
        --row_id="\$ROW_ID" \\
        --output_dir="$RESULTS_DIR" \\
        --dataset_name="$DATASET_NAME"
else
    echo "Skipping row \$ROW_ID as it exceeds $END."
fi
EOT
        )

        if [[ "$submission_output" =~ ^[0-9]+$ ]]; then
            echo "Chunk $i submitted with job ID $submission_output"
            chunk_job_ids+=("$submission_output")  # Store job ID
            break
        else
            echo "Submission issue: ${submission_output##*: } - Retrying in 5 minutes..."
            sleep 300
        fi
    done
done

echo "All $NUM_CHUNKS chunks submitted (queue limit: $MAX_JOBS_IN_QUEUE)."

# Wait for all chunk jobs to complete (initial submission)
echo "Waiting for all chunk jobs to complete..."
echo "Monitor progress in $RESULTS_DIR/logs/out.* and $RESULTS_DIR/logs/err.*"

# show how many completed in background every 15 minutes
while true; do
    completed_jobs=$(find "$RESULTS_DIR" -name "results_*.json" -exec grep -l '"Test_Loss"' {} + | wc -l)
    echo "$completed_jobs/$TOTAL_ROWS jobs completed successfully so far."
    sleep 900
done &
progress_pid=$!

for job_id in "${chunk_job_ids[@]}"; do
    while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
        sleep 30
    done
done

# Function to find which rows don't have a results file yet
incomplete_rows() {
    local missing_ids=()
    for ((rid=0; rid<"$TOTAL_ROWS"; rid++)); do
        if ! ls "${RESULTS_DIR}/results_${rid}_*.json" 1>/dev/null 2>&1; then
            missing_ids+=("$rid")
        fi
    done
    echo "${missing_ids[@]}"
}

MAX_RESUBMIT=5
attempt=1

while [ "$attempt" -le "$MAX_RESUBMIT" ]; do
    completed_jobs=$(find "$RESULTS_DIR" -name "results_*.json" -exec grep -l '"Test_Loss"' {} + | wc -l)
    if [ "$completed_jobs" -eq "$TOTAL_ROWS" ]; then
        echo "All $TOTAL_ROWS jobs completed successfully."
        break
    fi

    echo "Some rows did not complete. Attempt #$attempt at resubmitting..."
    missing_ids=( $(incomplete_rows) )
    if [ "${#missing_ids[@]}" -eq 0 ]; then
        echo "No incomplete rows found, but $completed_jobs/$TOTAL_ROWS have results. Stopping."
        break
    fi

    # Construct SLURM array range by grouping missing ids
    # e.g. 1-3,5,7 for missing rows [1,2,3,5,7], if you prefer
    # Here we'll just resubmit them as one array chunk:
    missing_min="${missing_ids[0]}"
    missing_max="${missing_ids[-1]}"
    
    submission_output=$(sbatch --parsable <<EOT 2>&1
#!/bin/bash
#SBATCH --job-name=retry_worker_${DATASET_NAME}_${attempt}
#SBATCH --array=${missing_min}-${missing_max}
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=${RESULTS_DIR}/logs/out.retry.%A_%a
#SBATCH --error=${RESULTS_DIR}/logs/err.retry.%A_%a
#SBATCH --killable
#SBATCH --requeue

source "$VENV_PATH/bin/activate"

ROW_ID=\${SLURM_ARRAY_TASK_ID}
if [[ " ${missing_ids[*]} " =~ " \$ROW_ID " ]]; then
    srun python "$WORKER_SCRIPT" \\
        --csv_file="$CSV_FILE" \\
        --row_id="\$ROW_ID" \\
        --output_dir="$RESULTS_DIR" \\
        --dataset_name="$DATASET_NAME"
else
    echo "Skipping row \$ROW_ID because it's already complete."
fi
EOT
    )

    echo "Retry submitted as job ID: $submission_output"
    
    # Wait for this retry job array to complete
    while squeue -j "$submission_output" 2>/dev/null | grep -q "$submission_output"; do
        sleep 30
    done
    
    attempt=$((attempt + 1))
done

# ----------------------
# Post-processing
# ----------------------

echo "Generating completion report..."
completed_jobs=$(find "$RESULTS_DIR" -name "results_*.json" -exec grep -l '"Test_Loss"' {} + | wc -l)
echo "$completed_jobs/$TOTAL_ROWS jobs completed successfully. See $RESULTS_DIR/logs for more details."

# Submit analysis job (depends on all completions)
if [ "$completed_jobs" -eq "$TOTAL_ROWS" ]; then
    echo "Submitting analysis job..."
    sbatch --dependency=afterok:$(echo "${chunk_job_ids[@]}" | tr ' ' ':') <<EOF
#!/bin/bash
#SBATCH --job-name=analysis_${DATASET_NAME}
#SBATCH --time=6:00:00
#SBATCH --output=${RESULTS_DIR}/analysis/analysis.out
#SBATCH --error=${RESULTS_DIR}/analysis/analysis.err
#SBATCH --gres=gpu:1,vmem:16G
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

source "$VENV_PATH/bin/activate"
python3 "$PROJECT_DIR/Parallel_run/analyze_results.py" \\
    --results_dir="$RESULTS_DIR" \\
    --dataset_name="$DATASET_NAME" \\
    --trans_dataset_name="$TRANS_DATASET_NAME"
EOF

    echo "Analysis job submitted. Monitor progress in ${RESULTS_DIR}/analysis.out"
    tail -F "${RESULTS_DIR}/analysis.out" "${RESULTS_DIR}/analysis.err" &
    tail_pid=$!
else
    echo "Not all jobs completed after $MAX_RESUBMIT retry attempts. Analysis job not submitted."
fi