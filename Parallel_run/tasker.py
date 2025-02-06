import sys
import torch
from Scripts.annotability_automations import *
from Datasets.dataset import *
from Datasets.merfish import Merfish
from Datasets.pbmc import PBMC
from Managers.anndata_manager import *

### GLOBAL PARAMETERS ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
repeats_per_size = 4
train_sizes = [1000]
### END GLOBAL PARAMETERS ###

### DATASET ###
dataset = PBMC()
dataset.load_data()
adata = dataset.preprocess_data()
adata = dataset.filter_by_health(clear_sick=True)
label_key = 'cell_type'
epoch_num_annot = 50
epoch_num_composition = 15
swap_probability = 0.1
percentile = 90
batch_size = 64
### END DATASET ###

format_manager = AnnDataManager()

format_manager.general_info(adata)
adata, group_counts = annotate("pbmc_healthy", adata, label_key, epoch_num_annot, 
                                    device, swap_probability, percentile, batch_size)
create_comps_for_workers(
    "pbmc_healthy", adata,
    train_sizes=train_sizes, repeats_per_size=repeats_per_size,
    )