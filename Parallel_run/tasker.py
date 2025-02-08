#!/usr/bin/env python3

import argparse
import logging
import torch
from Scripts.annotability_automations import *
from Datasets.dataset import *
from Datasets.merfish import Merfish
from Datasets.pbmc import PBMC
from Managers.anndata_manager import *
from Datasets.factory import get_dataset

def main():
    parser = argparse.ArgumentParser(description="Tasker script for dataset processing")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. pbmc_healthy)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Starting tasker for dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)
    format_manager.general_info(dataset.adata)
    adata = annotate(args.dataset, dataset.get_annotated_dataset(), dataset.label_key, dataset.epoch_num_annot, 
                                        device, dataset.swap_probability, dataset.percentile, dataset.batch_size)
    create_comps_for_workers(
        args.dataset, adata,
        train_sizes=train_sizes, repeats_per_size=repeats_per_size,
        )

if __name__ == "__main__":
    ### GLOBAL PARAMETERS ###
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    repeats_per_size = 4
    train_sizes = [1000]
    format_manager = AnnDataManager()
    ### END GLOBAL PARAMETERS ###
    main()