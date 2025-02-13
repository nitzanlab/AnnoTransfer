#!/usr/bin/env python3

import argparse
import logging
import torch
from Scripts.annotatability_automations import *
from Datasets.factory import get_dataset

def main():
    parser = argparse.ArgumentParser(description="Tasker script for dataset processing")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. pbmc_healthy)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Starting tasker for dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)
    dataset.manager.general_info(dataset.adata)
    adata = annotate(args.dataset)
    create_comps_for_workers(
        args.dataset, adata,
        train_sizes=train_sizes, 
        repeats_per_size=repeats_per_size,
        include_hard=True
        )

if __name__ == "__main__":
    ### GLOBAL PARAMETERS ###
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    repeats_per_size = 4
    train_sizes = [1000]
    ### END GLOBAL PARAMETERS ###
    main()