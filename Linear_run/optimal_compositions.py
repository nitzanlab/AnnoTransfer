import sys
import torch
from Scripts.annotatability_automations import *
from Datasets.dataset import *
from Datasets.merfish import Merfish
from Datasets.pbmc import PBMC
from Managers.anndata_manager import *
from Datasets.factory import get_dataset

### GLOBAL PARAMETERS ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
repeats_per_size = 3
train_sizes = [500, 750]
### END GLOBAL PARAMETERS ###

# Read the data argument from the command line
if len(sys.argv) < 3:
    print("Usage: python script.py <data>")
    sys.exit(1)

dataset_name = sys.argv[1]
trans_dataset_name = sys.argv[2]

# Hyperparameters for the dataset
dataset = get_dataset(dataset_name)
trans_dataset = get_dataset(trans_dataset_name)
adata = annotate(dataset_name)
label_key = dataset.label_key
epoch_num_annot = dataset.epoch_num_annot
epoch_num_composition = dataset.epoch_num_composition
swap_probability = dataset.swap_probability
percentile = dataset.percentile
batch_size = dataset.batch_size
format_manager = dataset.manager

format_manager.general_info(adata)
best_compositions, label_encoder = find_optimal_compositions(dataset_name, adata, label_key, train_sizes, 
                        repeats_per_size, device, epoch_num_composition, batch_size, format_manager)

for T in train_sizes:
    best_entry = min(best_compositions[T], key=lambda x: x["Test_Loss"])
    E, A, H = best_entry['composition']
    group_counts = {'Easy-to-learn': E, 'Ambiguous': A, 'Hard-to-learn': H}
    # compare dataset with optimal composition subset to not optimal composition subset and to full dataset
    comp_opt_subset_to_not(dataset_name, adata, label_key, group_counts, epoch_num_composition, epoch_num_annot, batch_size, format_manager)
    # compare again, this time on the dataset we're transferring to using the same optimal composition subset
    comp_opt_subset_to_not(trans_dataset_name, adata, label_key, group_counts, epoch_num_composition, epoch_num_annot, batch_size, format_manager)
