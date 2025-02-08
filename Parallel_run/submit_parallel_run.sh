job_id=$(sbatch $PROJECT_DIR/Parallel_run/tasker_and_dispatcher.sh | awk '{print $4}') && \
echo "Tasker job submitted with ID: $job_id" && \
tail -F main_controller.out main_controller.err