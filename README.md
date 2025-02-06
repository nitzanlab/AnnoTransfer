# AnnoTransfer
Utilizing compositions according to the Annotatability model to detect optimal subsets to be used for transfer learning.

## Tips
- The task of iteratively searching through all compositions is an expensive one that takes long. It's therefore recommended to run the tasks via the optimized .sh scripts available, which create sbatch jobs rather than .ipynb notebooks.

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
```
It will conveniently be used by all .sh scripts to determine the location of your installations in a single place.
Replace `<annoTransfer_installation>` with the **full path** to the location you cloned the library to (click `I` to enter edit mode, `esc` to quit mode).
Other paths can remain as they are if you're not making additional changes.
(Use `:wq` to save the file and exit)
From this point, `$PROJECT_DIR` refers to the path you set. It's recommended you add it to your environment for convenience.
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
2. On a machine with SLURM (such as phoenix) run `$PROJECT_DIR/Parallel_run/tasker_and_dispatcher.sh`.
#### 4.2 Linear Run
In a linear run, each composition will start training and reporting loss only once the one that preceded it completed.
1. Edit the global parameter if you wish in `$PROJECT_DIR/Linear_run/optimal_compositions.sbatch`. Here you can control batch size, subsets size, repeats, the dataset used etc.
2. If a machine with SLURM is available (such as phoenix) run `$PROJECT_DIR/Linear_run/optimal_compositions.sbatch`.
Otherwise, run `$PROJECT_DIR/Linear_run/optimal_compositions.py` directly.
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
