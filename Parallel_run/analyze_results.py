#!/usr/bin/env python3
import argparse
import json
import os
import glob
import pandas as pd

from collections import defaultdict
from Scripts.annotatability_automations import comp_opt_subset_to_not
from Datasets.factory import get_dataset

def load_results(results_dir):
    """Load all result JSONs into a DataFrame"""
    files = glob.glob(os.path.join(results_dir, "results_*.json"))
    records = []
    
    for f in files:
        with open(f) as fd:
            data = json.load(fd)
            if data.get("Test_Loss") is not None:
                records.append({
                    "Train_Size": data["Train_Size"],
                    "Easy": data["Easy"],
                    "Ambiguous": data["Ambiguous"],
                    "Hard": data["Hard"],
                    "Test_Loss": data["Test_Loss"]
                })
    
    return pd.DataFrame(records)

def determine_best_compositions(df):
    """Find best E/A/H composition for each train size"""
    best = defaultdict(dict)
    
    for train_size, group in df.groupby("Train_Size"):
        if not group.empty:
            best_row = group.loc[group["Test_Loss"].idxmin()]
            best[train_size] = {
                "composition": (best_row["Easy"], best_row["Ambiguous"], best_row["Hard"]),
                "loss": best_row["Test_Loss"]
            }
    
    return best

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--trans_dataset_name", required=True)
    args = parser.parse_args()

    # Load data
    dataset = get_dataset(args.dataset_name)
    trans_dataset = get_dataset(args.trans_dataset_name)
    
    # Process results
    df = load_results(args.results_dir)
    best_compositions = determine_best_compositions(df)
    
    # Run comparisons for each train size
    for train_size, data in best_compositions.items():
        E, A, H = data["composition"]
        group_counts = {"Easy-to-learn": E, "Ambiguous": A, "Hard-to-learn": H}
        
        print(f"Comparing for T={train_size} (E={E}, A={A}, H={H})")
        
        # compare dataset with optimal composition subset to not optimal composition subset and to full dataset
        comp_opt_subset_to_not(args.dataset_name, dataset.get_annotated_dataset(), dataset.label_key, group_counts, dataset.epoch_num_composition, dataset.epoch_num_annot, dataset.batch_size, dataset.manager)
        # compare again, this time on the dataset we're transferring to using the same optimal composition subset
        comp_opt_subset_to_not(args.trans_dataset_name, trans_dataset.get_annotated_dataset(), trans_dataset.label_key, group_counts, trans_dataset.epoch_num_composition, trans_dataset.epoch_num_annot, trans_dataset.batch_size, trans_dataset.manager)

if __name__ == "__main__":
    ### GLOBAL PARAMETERS ###
    max_epochs = 10
    ### END GLOBAL PARAMETERS ###

    main()