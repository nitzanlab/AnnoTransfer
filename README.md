# AnnoTransfer
Utilizing compositions according to the Annotatability model to detect optimal subsets to be used for transfer learning.

## Tips
- The task of iteratively searching through all compositions is an expensive one that takes long. It's therefore recommended to run the tasks via the optimized .sh scripts available, which create sbatch jobs rather than .ipynb notebooks.

## Getting Started
### 1.Clone the library. 
Sufficiently large space is required for many of the tasks and datasets. For phoenix cluster users, it's therefore recommended to clone it to their lab directory.
### 2.Set up configuration files
Create the following file in the path `~/.config/annoTransfer.conf`
It will conveniently be used by all .sh scripts to determine the location of your installations in a single place.
Replace `<annoTransfer_installation>` with the full path to the location you cloned the library to.
Other paths can remain as they are if you're not making additional changes.
```
# ~/.config/annoTransfer.conf
export PROJECT_DIR="<annoTransfer_installation>"  # Change this to your clone path
export VENV_PATH="$PROJECT_DIR/$VENV_NAME"
export VENV_NAME="annot_venv"
export TMP_DIR="$PROJECT_DIR/tmp"
export CACHE_DIR="$PROJECT_DIR/cache"
WORKDIR="$PROJECT_DIR"
```
From this point, `$(PROJECT_DIR)` refers to the path you set. It's recommended you add it to your environment for convenience.
### 3.Install virtual environment
Run `$(PROJECT_DIR)\Scripts\build_venv.sh`.
### 4.Run
#### 4.1 Linear Run
In a linear run, each composition will start training and reporting loss only once the one that preceded it completed.
1. Edit the global parameter as you wish in `$(PROJECT_DIR)\Linear_run\optimal_compositions.sbatch`.
2. If a machine with SLURM is available (such as phoenix) run `$(PROJECT_DIR)\Linear_run\optimal_compositions.sbatch`.
Otherwise, run `$(PROJECT_DIR)\Linear_run\optimal_compositions.py` directly.
#### 4.2 Parallel Run (recommended)
In a parallel run, first a csv will be created with all compositions required. Then, workers will be dispatched until all compositions results were reported to `Results` directory.
1. Edit the global parameter as you wish in `$(PROJECT_DIR)\Parallel_run\tasker.py`.
2. On a machine with SLURM (such as phoenix) run `$(PROJECT_DIR)\Parallel_run\tasker_and_dispatcher.sh`.
