#!/usr/bin/env python3

import argparse
import os
import json
import logging
import scanpy as sc

import torch
import pandas as pd

from Models.mlp_net import train_and_evaluate_mlp
from Datasets.pbmc import PBMC
from Managers.anndata_manager import AnnDataManager
from Scripts.annotability_automations import get_subset_composition
from Datasets.dataset import Dataset
from Datasets.factory import get_dataset

###############################################################################
# Worker function
###############################################################################
def worker_run_job(
    csv_file: str,
    row_id: int,
    output_dir: str,
    dataset_name: str,

):
    """
    Reads a single row from `csv_file` (pbmc_healthy_worker_jobs.csv),
    loads the pre-annotated PBMC dataset (pbmc_healthy_annotated.h5ad),
    uses get_subset_composition(...) and AnnDataManager to create train/test subsets,
    then runs train_and_evaluate_mlp(...) to get test_loss.

    Finally, writes the result to a per-job JSON file.
    """

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"[Worker] Starting worker_run_job, row_id={row_id}.")
    dataset = get_dataset(dataset_name)
    device = "cpu"
    epoch_num = dataset.epoch_num_composition
    batch_size = dataset.batch_size

    # -------------------------------------------------------------------------
    # 1) Read the job row from CSV
    # -------------------------------------------------------------------------
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"[Worker] CSV file '{csv_file}' does not exist.")

    jobs_df = pd.read_csv(csv_file)
    if row_id < 0 or row_id >= len(jobs_df):
        raise IndexError(f"[Worker] row_id={row_id} out of range (0..{len(jobs_df)-1}).")
    
    manager = AnnDataManager()
    job_row = jobs_df.iloc[row_id]

    train_size = int(job_row["Train_Size"])
    e = int(job_row["Easy"])
    a = int(job_row["Ambiguous"])
    h = int(job_row["Hard"])
    test_indices_str = job_row["Test_Indices"]
    run_id = int(job_row.get("Run", 1))
    out_file = "results_{train_size}_{run_id}_{e}_{a}_{h}.json"

    # if the output file already exists, skip this job
    out_path = os.path.join(output_dir, out_file)
    if os.path.exists(out_path):
        logging.warning(f"[Worker] Output file '{out_path}' already exists. Skipping.")
        return

    # Initialize test_loss and final_train_indices to safe defaults
    test_loss = None
    final_train_indices = []

    logging.info(f"[Worker] row={row_id} => T={train_size}, E={e}, A={a}, H={h}, run={run_id}")

    # Parse test indices
    test_indices = test_indices_str.split(",") if isinstance(test_indices_str, str) else []
    test_indices = [idx.strip() for idx in test_indices if idx.strip()]

    # -------------------------------------------------------------------------
    # 2) Load the (pre-annotated) PBMC dataset
    # -------------------------------------------------------------------------
    adata = dataset.get_annotated_dataset()
    label_key = dataset.label_key
    logging.info(f"[Worker] Loaded PBMC dataset shape: {adata.shape}")

    # Subset test data
    test_adata = manager.subset(adata, test_indices)

    # -------------------------------------------------------------------------
    # 3) Exclude the test set to form a "trainable" subset
    # -------------------------------------------------------------------------
    all_obs_index = set(adata.obs.index)
    test_set = set(test_indices)
    trainable_indices = list(all_obs_index - test_set)

    if len(trainable_indices) < train_size:
        logging.warning(
            f"[Worker] Not enough trainable samples for T={train_size}; have {len(trainable_indices)}. Skipping."
        )
    else:
        # Subset to only trainable indices
        adata_trainable = manager.subset(adata, trainable_indices)

        # Dictionary for how many "Easy", "Ambiguous", "Hard" we want
        group_counts_dict = {
            "Easy-to-learn": e,
            "Ambiguous": a,
            "Hard-to-learn": h
        }

        # ---------------------------------------------------------------------
        # 4) Get the actual train subset from adata_trainable
        # ---------------------------------------------------------------------
        try:
            train_adata, final_train_indices = get_subset_composition(adata_trainable, group_counts_dict)
            logging.info(f"[Worker] get_subset_composition succeeded: got {len(final_train_indices)} indices.")

            # Only proceed if we actually have the correct number
            if len(final_train_indices) == train_size:
                # -----------------------------------------------------------------
                # 5) Run training
                # -----------------------------------------------------------------
                device_torch = torch.device(device if torch.cuda.is_available() else "cpu")
                label_encoder = manager.getLabelEncoder(adata, label_key)

                try:
                    test_loss = train_and_evaluate_mlp(
                        adata_train=train_adata,
                        adata_test=test_adata,
                        label_key=label_key,
                        label_encoder=label_encoder,
                        num_classes=len(label_encoder.classes_),
                        epoch_num=epoch_num,
                        device=device_torch,
                        format_manager=manager,
                        batch_size=batch_size
                    )
                    logging.info(f"[Worker] row={row_id} => test_loss={test_loss}")
                except Exception as train_ex:
                    logging.error(f"[Worker] train_and_evaluate_mlp failed: {train_ex}")
                    test_loss = None
            else:
                logging.warning(f"[Worker] Mismatch: wanted train_size={train_size}, got {len(final_train_indices)}.")
        except Exception as ex:
            logging.error(f"[Worker] get_subset_composition failed: {ex}")

    # If training didn't happen or failed, test_loss remains None
    if test_loss is None:
        logging.warning(f"[Worker] has test_loss=None for row_id={row_id} due to errors or conditions. Exiting.")
        return

    # -------------------------------------------------------------------------
    # 6) Save the results
    # -------------------------------------------------------------------------

    result_data = {
        "row_id": row_id,
        "Train_Size": train_size,
        "Easy": e,
        "Ambiguous": a,
        "Hard": h,
        "Run": run_id,
        "Test_Indices": test_indices,
        "Train_Indices": final_train_indices,
        "Test_Loss": test_loss
    }

    try:
        with open(out_path, "w") as f:
            json.dump(result_data, f, indent=2)
        logging.info(f"[Worker] Saved results to {out_path}")
    except Exception as ex_save:
        logging.error(f"[Worker] Failed to save JSON: {ex_save}")

###############################################################################
# Main entry point
###############################################################################
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description="Worker script to process a single job from pbmc_healthy_worker_jobs.csv."
    )
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to the pbmc_healthy_worker_jobs.csv file.")
    parser.add_argument("--row_id", type=int, required=True,
                        help="0-based row index in the CSV.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save JSON files.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Directory to save JSON files.")

    args = parser.parse_args()
    worker_run_job(
        csv_file=args.csv_file,
        row_id=args.row_id,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )

if __name__ == "__main__":
    main()