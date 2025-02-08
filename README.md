# AnnoTransfer
Utilizing compositions according to the Annotatability model to detect optimal subsets to be used for transfer learning.

## Tips
- The task of iteratively searching through all compositions is an expensive one that takes long. It's therefore recommended to run the tasks via the optimized .sh scripts available, which create sbatch jobs rather than .ipynb notebooks.
- The provided datasets were preprocessed according to the authours discertion. Make sure to check that they match your excpectations. See `Datasets/<dataset name>.py`.

## Getting Started
### 1.Clone the library. 
Sufficiently large space is required for many of the tasks and datasets. For phoenix cluster users, it's therefore recommended to clone it to their lab directory.
### 2.Set up configuration files
Use `vi ~/.config/annoTransfer.conf` and paste the following into the newly created file:
```
# ~/.config/annoTransfer.conf
export PROJECT_DIR="<annoTransfer_installation>"  # Change this to your clone path
export VENV_NAME="annot_venv"
export VENV_PATH="$PROJECT_DIR/$VENV_NAME"
export TMP_DIR="$PROJECT_DIR/tmp"
export CACHE_DIR="$PROJECT_DIR/cache"
WORKDIR="$PROJECT_DIR"
export DATASET_NAME="pbmc_healthy" # change if want work on a different one
export TRANS_DATASET_NAME="pbmc_sick" # change if want to work on a different one
```

Replace `<annoTransfer_installation>` with the **full path** to the location you cloned the library to (click `I` to enter edit mode, `esc` to quit mode).
Other paths can remain as they are if you're not making additional changes
(type `:wq` to save the file and exit).

Now run `source ~/.config/annoTransfer.conf` to apply the variables to your enviornment. Run that command on every session you wish to acess these variables. (scripts in the library use them by default).
### 3.Install virtual environment
Run `$PROJECT_DIR/Scripts/build_venv.sh`.
### 4.Run
We we'll demonstrate running on a PBMC CVID dataset. To obtain it, run:
```
wget -O $PROJECT_DIR/datasets/pbmc_cvid.h5ad "https://datasets.cellxgene.cziscience.com/dfb51f99-a306-4daa-9f4a-afe7de65bbf2.h5ad"
```
For more details on datasets, see relevant section.
#### 4.1 Parallel Run (recommended)
In a parallel run, first a csv will be created with all compositions required. Then, workers will be dispatched until all compositions results were reported to `Results` directory.
1. Edit the global parameter or the used dataset if you wish in `$PROJECT_DIR/Parallel_run/tasker.py`. Here you can control batch size, subsets size, repeats, the dataset used etc.
2. On a machine with SLURM (such as phoenix) run `$PROJECT_DIR/Parallel_run/submit_parallel_run.sh`.

Note this can still take **very** long time, espicially for the PBMC dataset configured by default. 
However the script you just ran starts an up to days-long job on the cluster, so whenever you log out you can log back in and check on the progress. Keep the id provided in `Tasker job submitted with ID: <job_id>` to check on it later.
For details and help see Parallel Run section.
#### 4.2 Linear Run
In a linear run, each composition will start training and reporting loss only once the one that preceded it completed.
1. Edit the global parameter if you wish in `$PROJECT_DIR/Linear_run/optimal_compositions.sbatch`. Here you can control batch size, subsets size, repeats, the dataset used etc.
2. If a machine with SLURM is available (such as phoenix) run `$PROJECT_DIR/Linear_run/optimal_compositions.sbatch`.
Otherwise, run `$PROJECT_DIR/Linear_run/optimal_compositions.py` directly.

Only recommended for small datasets and forgiving compositions constraints.
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
Parallel run can conviniently be executed using a single shell script `Parallel_run/tasker_and_dispatcher.sh`. It consists of two stages exaplained below.

1. Creating the tasks that should be preformed. Implemented in `Parallel_run/tasker.py` and has two parts:

   1.1. The dataset is annotated with the easy, ambiguous and hard to learn using the `annotate` func in the `Scripts/annotability_automations.py`. This process can be several hours long for a dataset the size of PBMC CVID, even only for healthy patients, so the annoated vestion will be saved for future runs as `<dataset_name>_annotated.h5ad`. Any time `annotate` is called, it will first look for the annotated version to save time.

   1.2. Creating a csv where each row is a single composition (i.e. easy 10%, ambigious 80%, hard 10%) that should be trained and tested. These are the 'tasks'. The csv's name is defined as an enviornment variable. The function implementing this is `create_comps_for_workers` under `Scripts/annotability_automations.py`.
2. Submitting workers to execute each of the tasks in csv. The dispatcher calling the wotker can be found in `Parallel_run/tasker_and_dispatcher.sh`. And the script each such worker executes is implemented in `Parallel_run/worker_script.py`.

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
