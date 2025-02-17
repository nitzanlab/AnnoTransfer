# AnnoTransfer
Utilizing compositions according to the Annotatability model to detect optimal subsets to be used for transfer learning.

## Tips
- The task of iteratively searching through all compositions is an expensive one that takes long. It's therefore recommended to run the tasks via the optimized .sh scripts available, which create sbatch jobs rather than .ipynb notebooks, which could later be used for evaluation.
- For a parallel run (see relevant section) you also have the option to not consider composition that include HARD examples at all, greatly reducing computation time. To do so, set `include_hard=False` in `Parallel_run/tasker.py`.
- The provided datasets were preprocessed according to the author's discretion. Make sure to check that they match your excpectations. See `Datasets/<dataset name>.py`.

## Getting Started
### 0. Local introduction
To see locally what the library can do, follow the pre-compiled `example.ipynb` notebook. What follows is a comprehensive guide for running optimal compositions search on a computationally strong remote (i.e. phoenix for HUJI users) and all actions should be done on the remote.
### 1. Clone the library. 
Sufficiently large space is required for many of the tasks and datasets. For phoenix cluster users, it's therefore recommended to clone it to their lab directory.
### 2. Set up configuration files
Use `vi ~/.config/annoTransfer.conf` and paste the following into the newly created file:
```
# ~/.config/annoTransfer.conf
export PROJECT_DIR="<annoTransfer_installation>"  # Change this to your clone path
export VENV_NAME="annot_venv"
export VENV_PATH="$PROJECT_DIR/$VENV_NAME"
export TMP_DIR="$PROJECT_DIR/tmp"
export CACHE_DIR="$PROJECT_DIR/cache"
export DATASET_NAME="pbmc_healthy" # change if want work on a different one
export TRANS_DATASET_NAME="pbmc_sick" # change if want to work on a different one
```

Replace `<annoTransfer_installation>` with the **full path** to the location you cloned the library to (click `I` to enter edit mode, `esc` to quit mode).
Other paths can remain as they are if you're not making additional changes
(type `:wq` to save the file and exit).

Now run `source ~/.config/annoTransfer.conf` to apply the variables to your environment. Run that command on every session you wish to acess these variables. (scripts in the library use them by default).
### 3. Install virtual environment
Run `$PROJECT_DIR/Scripts/build_venv.sh`.
### 4. Run
We we'll demonstrate running on a PBMC CVID dataset. To obtain it, run:
```
wget -O $PROJECT_DIR/datasets/pbmc_cvid.h5ad "https://datasets.cellxgene.cziscience.com/dfb51f99-a306-4daa-9f4a-afe7de65bbf2.h5ad"
```
For more details on datasets, see relevant section.

Next, you can choose either to run all compositions on a single strong GPU sequentially through 4.2 linear run, or simultaneously through 4.1 Parallel run on many CPUs rather than GPUs. It's hard to tell upfront which is better, and 4.1 is heavily dependent on the remote's availability. You can always try both.

#### 4.1 Parallel Run
In a parallel run, first a csv will be created with all compositions required. Then, workers will be dispatched until all compositions results were reported to `Results` directory.
1. Edit the global parameter or the used dataset if you wish in `$PROJECT_DIR/Parallel_run/tasker.py`. Here you can control batch size, subsets size, repeats, the dataset used etc.
2. On a machine with SLURM (such as phoenix) run `$PROJECT_DIR/Parallel_run/submit_parallel_run.sh`. You can safely exit the script after seeing `tasker_and_dispatcher job submitted with ID` as it summoned the job in the background.

Tip: depending on the dataset at hand, you will need to adjust the video memory, GPU, CPUs, and physical memory used - though a more demending request will take longer before a node is available. You can make this change at the every section of `$PROJECT_DIR/Parallel_run/tasker_and_dispatcher.sh`. Each section (tasker, worker job, resubmission, analyzer) may benefit from tweaks appropriate to their compuatational demend. Some expermenetation may be required.

Note this can still take **very** long time.
However the script you just ran starts an up to days-long job on the cluster, so whenever you log out you can log back in and check on the progress. Keep the id provided in `Tasker job submitted with ID: <job_id>` to check on it later.
For details and help see Parallel Run section.
#### 4.2 Linear Run
In a linear run, each composition will start training and reporting loss only once the one that preceded it completed.
1. Edit the global parameter if you wish in `$PROJECT_DIR/Linear_run/optimal_compositions.py`. Here you can control batch size, subsets size, repeats, the dataset used etc.
2. If a machine with SLURM is available (such as phoenix) run `sbatch $PROJECT_DIR/Linear_run/test_compositions.sbatch`.
Otherwise, run `$PROJECT_DIR/Linear_run/optimal_compositions.py` directly.
Tip for SLURM users: depending on the dataset at hand, you will need to adjust the video memory, GPU, and physical memory used - though a more demending request will take longer before a node is available. You can make this change at the top of the `$PROJECT_DIR/Linear_run/test_compositions.sbatch` file.

## Datasets
Each Dataset should configured according to the interface determined in `Datasets/dataset.py`.
Two existing example for implementation can be found in:
- `Datasets/merfish.py`
- `Datasets/pbmc.py`

(If haven't done so already - run 
```
wget -O $PROJECT_DIR/datasets/pbmc_cvid.h5ad "https://datasets.cellxgene.cziscience.com/dfb51f99-a306-4daa-9f4a-afe7de65bbf2.h5ad"
```
to get the PBMC dataset used by the implementation.
)

New datasets should be added to the same folder and follow the same convention.
Any extra function can be incorporated as well in the dataset's .py file - PBMC allows to filter by sick and healthy patients for one via the `filter_by_health` func.
## Parallel Run
Parallel run can conveniently be executed using a single shell script `Parallel_run/tasker_and_dispatcher.sh`. It consists of the three stages explained below.

1. Creating the tasks that should be preformed. Implemented in `Parallel_run/tasker.py` and has two parts:

   1.1. The dataset is annotated with the easy, ambiguous and hard to learn using the `annotate` func in the `Scripts/annotability_automations.py`. This process can be several hours long for a dataset the size of PBMC CVID, even only for healthy patients, so the annotated version will be saved for future runs as `<dataset_name>_annotated.h5ad`. Any time `annotate` is called, it will first look for the annotated version to save time.

   1.2. Creating a csv where each row is a single composition (i.e. easy 10%, ambiguous 80%, hard 10%) that should be trained and tested. These are the 'tasks'. The csv's name is defined as an environment variable. The function implementing this is `create_comps_for_workers` under `Scripts/annotability_automations.py`.
2. Submitting workers to execute each of the tasks in csv. The dispatcher calling the worker can be found in `Parallel_run/tasker_and_dispatcher.sh`. And the script each such worker executes is implemented in `Parallel_run/worker_script.py`.
3. An `Analyzer` script will collect all output from workers and compile it to determine best composition. It will then run comparations on it for both the original dataset and data you transfer to, as determined in `~/.config/annoTransfer.conf`. 
`Analyzer` can be found in `$PROJECT_DIR/Parallel_run/analyze_results.py` while the comparator can be found in `$PROJECT_DIR/Parallel_run/annotability_automations.py` in func `comp_opt_subset_to_not`.
See `Results` subsection ahead for more info.


### Keeping track of the job
You can use any of the SLURM command using the job_id provided at the start of the run with `Tasker job submitted with ID: <job_id>`. 

To see the logs use :
```
cat $PROJECT_DIR/main_controller_<job_id>.out $PROJECT_DIR/main_controller_<job_id>.err
```
To start following them again:
```
tail -F $PROJECT_DIR/main_controller_<job_id>.out $PROJECT_DIR/main_controller_<job_id>.err
```
To check if the job still runs and more details:
```
scontrol show job <job_id>
```

### Results
The results will be provided at the end of the `tasker and dispatcher` script as part of the log. see above section to access it.
It is compiled and can be reproduced from the following:
- Under `$PROJECT_DIR/results_${DATASET_NAME}_$<script start time>` each worker's output will be available.
- Under `$PROJECT_DIR/results_${DATASET_NAME}_$<script start time>\analysis` the final analyzer's log will be available, and likely the results.
- Under `$PROJECT_DIR/results_${DATASET_NAME}_$<script start time>\logs` the worker's log can be found, for troubleshooting.
