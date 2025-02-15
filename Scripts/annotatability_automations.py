import warnings

warnings.filterwarnings("ignore")
import os
import sys
print(sys.executable)
print('\n'.join(sys.path))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.sparse as sp
from Annotatability import models
import logging
import contextlib
import io
import scanpy as sc
import torch.nn as nn
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Models.mlp_net import *
from Managers.anndata_manager import *
import csv
import json
from Datasets.factory import get_dataset
import seaborn as sns

#TODO: get rid of all label_encoder as inputs and use the dedicated function in the manager

model_runners = {
    "mlp":train_and_evaluate_mlp, 
    }

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("scvi").setLevel(logging.WARNING)

SMALL_SIZE = 16
MEDIUM_SIZE = 20.5
BIGGER_SIZE = 24

# Define custom color palette
annotation_order = ['Easy-to-learn', 'Ambiguous', 'Hard-to-learn']
annotation_colors = ['green', 'orange', 'red']
palette = dict(zip(annotation_order, annotation_colors))

# Initialize StringIO object to suppress outputs
f = io.StringIO()

def train_and_get_prob_list(adata, label_key, epoch_num, device, batch_size):
    logging.info('Training the model...')
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            prob_list = models.follow_training_dyn_neural_net(
                adata,
                label_key=label_key,
                iterNum=epoch_num,
                device=device,
                batch_size=batch_size
            )
    logging.info('Training complete.')
    return prob_list

def calculate_confidence_and_variability(prob_list, n_obs, epoch_num):
    logging.info('Calculating confidence and variability...')
    with contextlib.redirect_stdout(f):
        all_conf, all_var = models.probability_list_to_confidence_and_var(
            prob_list,
            n_obs=n_obs,
            epoch_num=epoch_num
        )
    logging.info('Calculation complete.')
    return all_conf, all_var

def find_cutoffs(adata, label_key, device, probability, percentile, epoch_num):
    logging.info('Finding cutoffs...')
    with contextlib.redirect_stdout(f):
        cutoff_conf, cutoff_var = models.find_cutoff_paramter(
            adata,
            label_key,
            device,
            probability=probability,
            percentile=percentile,
            epoch_num=epoch_num
        )
    logging.info('Cutoffs found: cutoff_conf=%s, cutoff_var=%s', cutoff_conf, cutoff_var)
    return cutoff_conf, cutoff_var

def assign_annotations(adata, all_conf, all_var, cutoff_conf, cutoff_var, annotation_col='Annotation'):
    logging.info('Assigning annotations...')
    adata.obs["var"] = all_var.detach().numpy()
    adata.obs["conf"] = all_conf.detach().numpy()
    adata.obs['conf_binaries'] = pd.Categorical(
        (adata.obs['conf'] > cutoff_conf) | (adata.obs['var'] > cutoff_var)
    )

    annotation_list = []
    # Disable tqdm output by setting disable=True
    for i in tqdm(range(adata.n_obs), desc='Assigning annotations', disable=True):
        if adata.obs['conf_binaries'].iloc[i]:
            if (adata.obs['conf'].iloc[i] > 0.95) & (adata.obs['var'].iloc[i] < 0.15):
                annotation_list.append('Easy-to-learn')
            else:
                annotation_list.append('Ambiguous')
        else:
            annotation_list.append('Hard-to-learn')

    adata.obs[annotation_col] = annotation_list
    adata.obs['Confidence'] = adata.obs['conf']
    adata.obs['Variability'] = adata.obs['var']
    logging.info('Annotation assignment complete.')
    return adata

def annotate(dataset_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logging.info('Starting annotation process...')
    dataset = get_dataset(dataset_name)
    
    try: 
        adata = dataset.get_annotated_dataset()
        logging.info('Loaded existing annotated dataset.')
        return adata
    
    except FileNotFoundError:
        adata = dataset.adata  # Use the non-annotated dataset
        prob_list = train_and_get_prob_list(adata, label_key=dataset.label_key, epoch_num=dataset.epoch_num_annot, 
                                        device=device, batch_size=dataset.batch_size)
        all_conf, all_var = calculate_confidence_and_variability(prob_list, n_obs=adata.n_obs, 
                                                                    epoch_num=dataset.epoch_num_annot)
        conf_cutoff, var_cutoff = find_cutoffs(adata, dataset.label_key, device, 
                                                probability=dataset.swap_probability, 
                                                percentile=dataset.percentile, 
                                                epoch_num=dataset.epoch_num_annot)
        adata = assign_annotations(adata, all_conf, all_var, conf_cutoff, var_cutoff, 
                                    annotation_col='Annotation')
        adata.write(dataset_name + '_annotated.h5ad')
        group_counts = adata.obs['Annotation'].value_counts()
        logging.info('Annotation process complete.')
        logging.info('Group counts: %s', group_counts.to_dict())
        return adata

def find_optimal_compositions(
    dataset_name,
    adata,
    label_key,
    train_sizes,
    repeats_per_size,
    device,
    epoch_num,
    batch_size,
    format_manager,
    model="mlp"
):
    """
    Runs the training and evaluation experiment for a given dataset.

    Parameters:
    - dataset_name (str): Name identifier for the dataset (e.g., 'merfish', 'pbmc').
    - adata (AnnData): The dataset to process.
    - label_key (str): The key in adata.obs that contains the labels.
    - train_sizes (list of int): List of training set sizes to experiment with.
    - repeats_per_size (int): Number of repeats for each training size.
    - device (torch.device): The device to run the training on ('cpu' or 'cuda').
    - epoch_num (int): Number of training epochs.
    - batch_size (int): Batch size for training.

    Returns:
    - best_compositions (dict): Dictionary containing the best compositions and their corresponding test losses.
    """
    logging.info('Starting find_optimal_compositions for dataset: %s', dataset_name)
    csv_file = dataset_name + '_optimal_compositions.csv'

    # Load existing results from CSV or create an empty DataFrame
    try:
        logging.info('Loading existing results from %s', csv_file)
        results_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        logging.warning('CSV file %s not found. Starting with an empty DataFrame.', csv_file)
        # Include 'Train_Indices' and 'Test_Indices' columns
        results_df = pd.DataFrame(columns=['Train_Size', 'Easy', 'Ambiguous', 'Hard', 'Test_Loss', 'Train_Indices', 'Test_Indices'])

    # Convert the 'Train_Size' column to a dictionary with counts for faster lookup
    existing_counts = results_df['Train_Size'].value_counts().to_dict()
    logging.debug('Existing counts per Train_Size: %s', existing_counts)

    # Assuming 'group_counts' is a pandas Series with annotations as indices
    group_counts = adata.obs['Annotation'].value_counts()
    logging.info('Group counts in the data: %s', group_counts.to_dict())

    # Assign counts to E, A, H
    E = group_counts.get('Easy-to-learn', 0)
    A = group_counts.get('Ambiguous', 0)
    H = group_counts.get('Hard-to-learn', 0)

    # Get the indices of each group
    easy_indices = adata.obs.index[adata.obs['Annotation'] == 'Easy-to-learn'].tolist()
    ambiguous_indices = adata.obs.index[adata.obs['Annotation'] == 'Ambiguous'].tolist()
    hard_indices = adata.obs.index[adata.obs['Annotation'] == 'Hard-to-learn'].tolist()

    # Fit LabelEncoder on the entire dataset labels
    label_encoder = LabelEncoder()
    label_encoder.fit(adata.obs[label_key])
    num_classes = len(label_encoder.classes_)

    best_compositions = {}

    for T in train_sizes:
        current_runs = existing_counts.get(T, 0)
        runs_needed = repeats_per_size - current_runs


        if runs_needed <= 0:
            # Use existing entries
            existing_rows = results_df[results_df['Train_Size'] == T]
            for idx, row in existing_rows.iterrows():
                easy = row['Easy']
                ambiguous = row['Ambiguous']
                hard = row['Hard']
                test_loss = row['Test_Loss']
                train_indices_str = row.get('Train_Indices', None)
                test_indices_str = row.get('Test_Indices', None)
                logging.info(
                    f"Using cached result for {dataset_name} Train_Size={T}: Easy={easy}, Ambiguous={ambiguous}, Hard={hard}, Test Loss={test_loss}"
                )
                # Store the cached results
                if T not in best_compositions:
                    best_compositions[T] = []
                best_compositions[T].append({
                    'composition': (easy, ambiguous, hard),
                    'Test_Loss': test_loss,
                    'Train_Indices': train_indices_str,
                    'Test_Indices': test_indices_str
                })
            continue  # Skip computation for this T as all repeats are already done

        else:
            # Calculate test size (25% of train size)
            test_size = int(0.25 * T)
            total_size = T + test_size

            # Select the test indices once per dataset size
            all_indices = adata.obs.index.tolist()
            # Ensure we have enough samples for test set
            if len(all_indices) < test_size:
                logging.warning(f"Not enough samples for Test Size={test_size} at Train_Size={T}")
                continue  # Skip if not enough samples

            # Randomly sample test_size samples for the test set
            test_indices = random.sample(all_indices, test_size)

            # Define step size as a function of T
            step_size = max(1, T // 100)

            # Generate compositions summing up to T (train size)
            compositions = []
            E = group_counts.get('Easy-to-learn', 0)
            A = group_counts.get('Ambiguous', 0)
            H = group_counts.get('Hard-to-learn', 0)
            logging.debug('Group counts: E=%d, A=%d, H=%d', E, A, H)
            for e in range(0, min(T, E) + 1, step_size):
                for a in range(0, min(T - e, A) + 1, step_size):
                    h = T - e - a
                    if h >= 0 and h <= H:
                        compositions.append((e, a, h))
            if not compositions:
                logging.warning(f"No valid compositions for Train Size={T}")
                # Save an entry indicating no valid compositions
                new_row = {
                    'Train_Size': T,
                    'Easy': None,
                    'Ambiguous': None,
                    'Hard': None,
                    'Test_Loss': None,
                    'Train_Indices': None,
                    'Test_Indices': ','.join(map(str, test_indices))
                }
                new_row_df = pd.DataFrame([new_row])
                results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                results_df.to_csv(csv_file, index=False)
                continue
            
            logging.info(f"Found {len(compositions)} valid compositions for Train Size={T}, starting runs...")
            for run in range(current_runs + 1, repeats_per_size + 1):
                min_loss = float('inf')
                best_comp = None
                best_train_indices = None

                # For each composition, train and get test loss
                # Disable tqdm output by setting disable=True
                for comp in tqdm(compositions, desc=f"Testing compositions for Train Size={T} - Run {run}", disable=True):
                    e, a, h = comp
                    # Ensure not exceeding group counts
                    if e > E or a > A or h > H:
                        continue  # Invalid composition

                    # Ensure we have enough samples in each group
                    if len(easy_indices) < e or len(ambiguous_indices) < a or len(hard_indices) < h:
                        continue  # Skip if not enough samples

                    # Randomly sample e, a, h samples from each group for training
                    available_easy = list(set(easy_indices) - set(test_indices))
                    available_ambiguous = list(set(ambiguous_indices) - set(test_indices))
                    available_hard = list(set(hard_indices) - set(test_indices))

                    if len(available_easy) < e or len(available_ambiguous) < a or len(available_hard) < h:
                        continue  # Not enough samples after excluding test set

                    train_easy_indices = random.sample(available_easy, e) if e > 0 else []
                    train_ambiguous_indices = random.sample(available_ambiguous, a) if a > 0 else []
                    train_hard_indices = random.sample(available_hard, h) if h > 0 else []
                    train_indices = train_easy_indices + train_ambiguous_indices + train_hard_indices

                    # Ensure total train samples equal T
                    if len(train_indices) != T:
                        continue  # Skip if train size mismatch

                    # Create training and testing datasets
                    adata_train = adata[train_indices].copy()
                    adata_test = adata[test_indices].copy()

                    train_and_eval = model_runners[model]
                    # Train and get test loss
                    test_loss = train_and_eval(
                        adata_train=adata_train, 
                        adata_test=adata_test, 
                        label_key=label_key, 
                        label_encoder=label_encoder,
                        num_classes=num_classes,
                        epoch_num=epoch_num, 
                        device=device, 
                        format_manager=format_manager,
                        batch_size=batch_size
                    )

                    # Update minimum loss and best composition
                    if test_loss < min_loss:
                        min_loss = test_loss
                        best_comp = comp
                        best_train_indices = train_indices.copy()

                    logging.info(f"Run {run} out of {repeats_per_size} for Train_Size={T}: Easy={e}, Ambiguous={a}, Hard={h}, Test Loss={test_loss}")

                if best_comp is not None:
                    easy, ambiguous, hard = best_comp

                    # Append to best_compositions
                    if T not in best_compositions:
                        best_compositions[T] = []
                    best_compositions[T].append({
                        'composition': best_comp,
                        'Test_Loss': min_loss,
                        'Train_Indices': best_train_indices,
                        'Test_Indices': test_indices  # Same test_indices for all runs of this T
                    })

                    # Save the result to the DataFrame and CSV
                    new_row = {
                        'Train_Size': T,
                        'Easy': easy,
                        'Ambiguous': ambiguous,
                        'Hard': hard,
                        'Test_Loss': min_loss,
                        'Train_Indices': ','.join(map(str, best_train_indices)),
                        'Test_Indices': ','.join(map(str, test_indices))
                    }
                    new_row_df = pd.DataFrame([new_row])
                    results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                    results_df.to_csv(csv_file, index=False)
                    logging.info(f"Run {run} out of {repeats_per_size} for Train_Size={T}: Easy={easy}, Ambiguous={ambiguous}, Hard={hard}, Test Loss={min_loss}")
                else:
                    logging.warning(f"No valid compositions found for {dataset_name} Train_Size={T} (Run {run})")
                    # Save an entry indicating no valid compositions
                    new_row = {
                        'Train_Size': T,
                        'Easy': None,
                        'Ambiguous': None,
                        'Hard': None,
                        'Test_Loss': None,
                        'Train_Indices': None,
                        'Test_Indices': ','.join(map(str, test_indices))
                    }
                    new_row_df = pd.DataFrame([new_row])
                    results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                    results_df.to_csv(csv_file, index=False)
                logging.info(f"Completed Run {run} out of {repeats_per_size} for Train_Size={T}")
    logging.info('find_optimal_compositions completed for dataset: %s', dataset_name)
    return best_compositions, label_encoder

def create_comps_for_workers(
    dataset_name,
    adata,
    train_sizes,
    repeats_per_size,
    include_hard=True
):
    """
    Create a CSV of all (Train_Size, composition, run) jobs needed, *without* running training.
    Each row can later be picked up by a worker script.

    Parameters:
    - dataset_name (str): Name identifier for the dataset (e.g., 'merfish', 'pbmc').
    - adata (AnnData): The dataset to process.
    - label_key (str): The key in adata.obs that contains the labels.
    - train_sizes (list of int): List of training set sizes to experiment with.
    - repeats_per_size (int): Number of repeats for each training size.
    - include_hard (bool): Whether to include Hard-to-learn samples in the compositions.

    This function does NOT run training. It only enumerates:
      1) A random test set for each Train_Size
      2) All valid (Easy, Ambiguous, Hard) splits that sum to Train_Size
      3) The repeated runs

    Then saves a CSV of jobs. Each row = one job to be run by a worker script.
    """
    logging.info('Starting find_optimal_compositions_using_workers for dataset: %s', dataset_name)

    # This is the CSV that will list the "jobs" we want each worker to run.
    csv_file = dataset_name + '_worker_jobs.csv'

    # If it already exists, do NOT overwrite
    if os.path.exists(csv_file):
        logging.info('CSV file %s already exists. Skipping creation.', csv_file)
        return

    # Re-derive group counts from the actual data
    obs_counts = adata.obs['Annotation'].value_counts()
    logging.info('Group counts in the data: %s', obs_counts.to_dict())

    # Extract counts for each group
    E_count = obs_counts.get('Easy-to-learn', 0)
    A_count = obs_counts.get('Ambiguous', 0)
    H_count = obs_counts.get('Hard-to-learn', 0)

    # We'll build rows for a DataFrame that mimics the original columns
    # plus "Run" (so we can handle repeats).
    job_rows = []

    # Full list of all sample indices (used for picking test sets)
    all_indices = adata.obs.index.tolist()

    # For each requested Train_Size
    for T in train_sizes:
        # The test set is 25% of T, chosen once per T
        test_size = int(0.25 * T)
        if len(all_indices) < test_size:
            logging.warning(f"Not enough samples for Test Size={test_size} at Train_Size={T}. Skipping.")
            continue

        # Randomly sample test_size from the entire dataset
        test_indices = random.sample(all_indices, test_size)
        test_indices_str = ",".join(map(str, test_indices))

        # step_size logic
        step_size = max(1, T // 100)

        # Generate all compositions (e,a,h) summing to T
        compositions = []
        for e in range(0, min(T, E_count) + 1, step_size):
            if include_hard:
                for a in range(0, min(T - e, A_count) + 1, step_size):
                    h = T - e - a
                    if 0 <= h <= H_count:
                        compositions.append((e, a, h))
            else:
                a = T - e
                h = 0
                if 0 <= a <= A_count:
                    compositions.append((e, a, h))

        if not compositions:
            logging.warning(f"No valid compositions for Train Size={T}. Skipping.")
            continue

        # For each repeat from 1..repeats_per_size
        for run_idx in range(repeats_per_size):
            # For each composition
            for (e, a, h) in tqdm(compositions,
                                    desc=f"Preparing job rows for T={T} (Run {run_idx+1})",
                                    disable=True):
                job_rows.append({
                    "Train_Size": T,
                    "Run": run_idx + 1,
                    "Easy": e,
                    "Ambiguous": a,
                    "Hard": h,
                    "Test_Indices": test_indices_str
                })

    # Convert to DataFrame
    columns = ["Train_Size", "Run", "Easy", "Ambiguous", "Hard", "Test_Loss",
                "Train_Indices", "Test_Indices"]
    jobs_df = pd.DataFrame(job_rows, columns=columns)

    # Save out to CSV
    jobs_df.to_csv(csv_file, index=False)
    logging.info("Created job CSV %s with %d rows of jobs.", csv_file, len(jobs_df))

    return jobs_df

def visualize_optimal_compositions_with_std(dataset_name):
    csv_file = dataset_name + '_optimal_compositions.csv'

    logging.info('Starting visualization of optimal compositions...')
    # Load the compositions from the CSV file
    try:
        results_df = pd.read_csv(csv_file)
        logging.info('Loaded results from %s', csv_file)
    except FileNotFoundError:
        logging.error(f"CSV file '{csv_file}' not found.")
        results_df = pd.DataFrame(columns=['Train_Size', 'Easy', 'Ambiguous', 'Hard', 'Test_Loss', 'Train_Indices', 'Test_Indices'])

    # Filter out rows with missing compositions
    results_df = results_df.dropna(subset=['Easy', 'Ambiguous', 'Hard'])

    # Convert counts to floats and Train_Size to int
    results_df['Easy'] = results_df['Easy'].astype(float)
    results_df['Ambiguous'] = results_df['Ambiguous'].astype(float)
    results_df['Hard'] = results_df['Hard'].astype(float)
    results_df['Train_Size'] = results_df['Train_Size'].astype(int)

    # Check if all train sizes have the same number of runs
    counts_per_size = results_df['Train_Size'].value_counts()
    if counts_per_size.nunique() != 1:
        logging.warning("Not all train sizes have the same number of rows in the CSV for each train size.")

    # Calculate total and proportions for each row
    results_df['Total'] = results_df['Easy'] + results_df['Ambiguous'] + results_df['Hard']
    results_df['Proportion_Easy'] = results_df['Easy'] / results_df['Total']
    results_df['Proportion_Ambiguous'] = results_df['Ambiguous'] / results_df['Total']
    results_df['Proportion_Hard'] = results_df['Hard'] / results_df['Total']

    # Group by Train_Size and calculate mean and std of proportions
    grouped = results_df.groupby('Train_Size').agg({
        'Proportion_Easy': ['mean', 'std'],
        'Proportion_Ambiguous': ['mean', 'std'],
        'Proportion_Hard': ['mean', 'std']
    }).reset_index()

    # Flatten MultiIndex columns
    grouped.columns = [
        'Train_Size',
        'Proportion_Easy_mean', 'Proportion_Easy_std',
        'Proportion_Ambiguous_mean', 'Proportion_Ambiguous_std',
        'Proportion_Hard_mean', 'Proportion_Hard_std'
    ]

    # Ensure that the mean proportions sum to 1 (approximately)
    if not np.allclose(grouped[['Proportion_Easy_mean', 'Proportion_Ambiguous_mean', 'Proportion_Hard_mean']].sum(axis=1), 1):
        logging.error("Mean proportions do not sum to 1.")
        raise ValueError("Mean proportions do not sum to 1.")

    # Prepare data for plotting
    train_sizes = grouped['Train_Size'].values
    proportion_e_mean = grouped['Proportion_Easy_mean'].values
    proportion_a_mean = grouped['Proportion_Ambiguous_mean'].values
    proportion_h_mean = grouped['Proportion_Hard_mean'].values
    proportion_e_std = grouped['Proportion_Easy_std'].values
    proportion_a_std = grouped['Proportion_Ambiguous_std'].values
    proportion_h_std = grouped['Proportion_Hard_std'].values

    # Verify that all arrays have the same length
    array_lengths = [
        len(train_sizes), len(proportion_e_mean), len(proportion_a_mean), len(proportion_h_mean),
        len(proportion_e_std), len(proportion_a_std), len(proportion_h_std)
    ]
    if len(set(array_lengths)) != 1:
        logging.error(f"Array length mismatch: {array_lengths}")
        raise ValueError(f"Array length mismatch: {array_lengths}")

    # Plotting grouped bar chart with error bars and preferred visualization properties
    logging.info('Creating plot for optimal compositions...')
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.25
    index = np.arange(len(train_sizes))

    # Plot the bars with the preferred colors and labels
    ax.bar(index - bar_width, proportion_e_mean, bar_width,
                       yerr=proportion_e_std, capsize=5, label='Easy-to-learn', color='green')
    ax.bar(index, proportion_a_mean, bar_width,
                            yerr=proportion_a_std, capsize=5, label='Ambiguous', color='orange')
    ax.bar(index + bar_width, proportion_h_mean, bar_width,
                       yerr=proportion_h_std, capsize=5, label='Hard-to-learn', color='red')

    # Customize the axes
    ax.set_xticks(index)
    ax.set_xticklabels([str(size) for size in train_sizes], rotation=45)
    ax.set_ylabel('Average Proportion')
    ax.set_xlabel('Train Set Size')
    ax.set_title('Optimal Composition of Train Set Samples with Standard Deviation')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_annot_std_chart.png')
    logging.info('Visualization saved as optimal_compositions.png')

def visualize_optimal_compositions_stacked_bar(dataset_name):
    """
    Loads the '_optimal_compositions.csv' file, then creates
    a stacked bar chart of Easy/Ambiguous/Hard counts vs. Train_Size.
    """

    csv_file = f"{dataset_name}_optimal_compositions.csv"
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found.")
        return

    # Load the compositions from the CSV file
    results_df = pd.read_csv(csv_file)

    # Drop rows with missing composition columns
    results_df = results_df.dropna(subset=['Easy', 'Ambiguous', 'Hard'])

    # Convert columns to float/int
    results_df['Easy'] = results_df['Easy'].astype(float)
    results_df['Ambiguous'] = results_df['Ambiguous'].astype(float)
    results_df['Hard'] = results_df['Hard'].astype(float)
    results_df['Train_Size'] = results_df['Train_Size'].astype(int)

    # Melt to reshape data for plotting
    df_melted = results_df.melt(
        id_vars=['Train_Size', 'Test_Loss'],
        value_vars=['Easy', 'Ambiguous', 'Hard'],
        var_name='Group',
        value_name='Count'
    )

    # Pivot to get stacked structure
    df_pivot = df_melted.pivot_table(
        index='Train_Size',
        columns='Group',
        values='Count',
        aggfunc='sum'
    ).fillna(0)

    # Define custom color palette
    palette = {
        'Easy': 'green',
        'Ambiguous': 'orange',
        'Hard': 'red'
    }

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Reorder columns for consistent stack order
    df_pivot = df_pivot[['Easy', 'Ambiguous', 'Hard']]

    ax = df_pivot.plot(
        kind='bar',
        stacked=True,
        color=[palette[group] for group in ['Easy', 'Ambiguous', 'Hard']],
        width=0.8,
        edgecolor='none',
        figsize=(12, 6)
    )

    ax.set_title(f'{dataset_name.upper()} Dataset: Optimal Group Composition vs. Train_Size')
    ax.set_ylabel('Number of Cells')
    ax.set_xlabel('Train_Size (Number of Cells)')
    ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(df_pivot.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_optimal_compositions_stacked_bar.png')
    plt.show()

def annot_bar_chart(best_compositions):
    df = pd.DataFrame(best_compositions)
    # print columns
    print(df.columns)

    # Prepare data for visualization
    df_melted = df.melt(
        id_vars=['Train_Size', 'Test_Loss'],
        value_vars=['Easy-to-learn', 'Ambiguous', 'Hard-to-learn'],
        var_name='Group',
        value_name='Count'
    )
    
    df_pivot = df_melted.pivot(
        index='Train_Size',
        columns='Group',
        values='Count'
    ).fillna(0)

    # Define custom color palette
    palette = {'Easy-to-learn': 'green', 'Ambiguous': 'orange', 'Hard-to-learn': 'red'}

    # Set the figure size and style
    plt.figure(figsize=(15, 7))
    sns.set_style("whitegrid")

    # Pivot the data for stacked plotting
    df_pivot = df_melted.pivot(
        index='Train_Size', columns='Group', values='Count'
    ).fillna(0)

    # Reorder the columns to match the group_labels order
    df_pivot = df_pivot[['Easy-to-learn', 'Ambiguous', 'Hard-to-learn']]

    # Plot the stacked bar chart
    ax = df_pivot.plot(
        kind='bar',
        stacked=True,
        color=[palette[group] for group in ['Easy-to-learn', 'Ambiguous', 'Hard-to-learn']],
        width=1.0,
        edgecolor='none',
        figsize=(15, 7)
    )

    # Adjust the plot
    ax.set_title('PBMC Dataset: Optimal Group Composition vs. Train_Size')
    ax.set_ylabel('Number of Cells')
    ax.set_xlabel('Train_Size (Number of Cells)')
    ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(df_pivot.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def highest_confidence_samples(adata, train_sizes, device, global_label_encoder, dataset_name, label_key):
    input_csv = dataset_name + '_optimal_compositions.csv'
    high_conf_csv = dataset_name + '_high_confidence_compositions.csv'

    logging.info('Starting processing of highest confidence samples...')
    # Read the csv to get the test indices used previously
    best_comp_df = pd.read_csv(input_csv)

    # Initialize a new DataFrame to store the results
    high_conf_df = pd.DataFrame(columns=['Train_Size', 'Train_Indices', 'Test_Indices', 'Test_Loss'])

    # Sort the samples by confidence scores in descending order
    # Assuming 'conf' is the column in adata.obs that contains confidence scores
    sorted_conf = adata.obs.sort_values(by='conf', ascending=False)

    # For each train size
    for T in train_sizes:
        logging.info(f"Processing high-confidence composition for Train_Size={T}")
        
        # Get the entries for Train_Size=T to retrieve the test indices
        size_df = best_comp_df[best_comp_df['Train_Size'] == T]
        
        if size_df.empty:
            logging.warning(f"No entries found for Train_Size={T} in {input_csv}. Skipping.")
            continue
        
        # Assuming all entries for a given Train_Size use the same Test_Indices
        # Fetch unique Test_Indices for Train_Size=T
        unique_test_indices = size_df['Test_Indices'].unique()
        
        if len(unique_test_indices) != 1:
            logging.warning(f"Multiple test sets found for Train_Size={T}. Using the first one.")
        
        test_indices_str = unique_test_indices[0]
        
        if pd.isnull(test_indices_str):
            logging.warning(f"No test indices found for Train_Size={T}. Skipping.")
            continue  # Skip if no test indices
        
        test_indices = test_indices_str.split(',')
        
        # Select the top T samples with highest confidence as the training set
        # Ensure no overlap with test_indices
        top_conf_indices = sorted_conf.index.difference(test_indices)[:T].tolist()
        
        if len(top_conf_indices) < T:
            logging.warning(f"Not enough available samples to select top {T} without overlapping with test set.")
            logging.warning(f"Available samples: {len(top_conf_indices)}")
            continue  # Skip if not enough samples
        
        # Create training and testing datasets
        adata_train = adata[top_conf_indices].copy()
        adata_test = adata[test_indices].copy()
        
        # Train and get test loss
        test_loss = train_and_evaluate_mlp(
            adata_train, adata_test, label_key, label_encoder=global_label_encoder,
            num_classes=len(global_label_encoder.classes_),  # Added num_classes
            epoch_num=30, device=device, batch_size=64, format_manager=AnnDataManager()
        )
        
        # Save the result using direct indexing (instead of append)
        high_conf_df.loc[len(high_conf_df)] = {
            'Train_Size': T,
            'Train_Indices': ','.join(map(str, top_conf_indices)),
            'Test_Indices': ','.join(map(str, test_indices)),
            'Test_Loss': test_loss
        }
        
        logging.info(f"Train_Size={T}, Test Loss={test_loss}")

    # Save the results to a new CSV file
    high_conf_df.to_csv(high_conf_csv, index=False)
    logging.info("High-confidence compositions have been saved to 'high_confidence_compositions.csv'.")

    # Read the results from the CSV files
    optimal_comp_df = pd.read_csv(input_csv)
    high_conf_df = pd.read_csv(high_conf_csv)

    # Calculate the average test loss for each Train_Size in the optimal compositions
    optimal_loss_df = optimal_comp_df.groupby('Train_Size')['Test_Loss'].mean().reset_index()
    optimal_loss_df.rename(columns={'Test_Loss': 'Optimal_Test_Loss'}, inplace=True)

    # Prepare the test loss for the high confidence compositions
    high_conf_loss_df = high_conf_df[['Train_Size', 'Test_Loss']]
    high_conf_loss_df.rename(columns={'Test_Loss': 'High_Conf_Test_Loss'}, inplace=True)

    # Merge the two DataFrames on 'Train_Size'
    comparison_df = pd.merge(optimal_loss_df, high_conf_loss_df, on='Train_Size')

    # Sort by Train_Size
    comparison_df.sort_values('Train_Size', inplace=True)

    # Plotting
    logging.info('Creating comparison plot...')
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_df['Train_Size'], comparison_df['Optimal_Test_Loss'], marker='o', label='Optimal Composition')
    plt.plot(comparison_df['Train_Size'], comparison_df['High_Conf_Test_Loss'], marker='s', label='High Confidence Composition')

    plt.title('Comparison of Test Losses by Train Size')
    plt.xlabel('Train Size')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dataset_name}_comparison_plot.png')
    logging.info('Comparison plot saved as comparison_plot.png')
    plt.show()

def train_validate_and_evaluate(
    train_dataset,
    validation_dataset,
    test_dataset,
    dataset_manager,
    label_key,
    label_encoder,
    num_classes,
    epoch_num,
    device,
    batch_size
):
    """
    Trains, validates, and evaluates a neural network model, selecting the best-performing epoch.

    Parameters:
    - train_dataset: Training dataset.
    - validation_dataset: Validation dataset.
    - test_dataset: Testing dataset.
    - dataset_manager: A manager object responsible for handling dataset-specific logic.
    - label_key (str): Key in the dataset containing the labels.
    - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
    - num_classes (int): Number of unique classes in the dataset.
    - epoch_num (int): Number of training epochs.
    - device (str or torch.device): Device to run the training on ('cpu' or 'cuda').
    - batch_size (int): Batch size for training.

    Returns:
    - best_epoch (int): The epoch with the highest validation performance.
    - test_loss (float): Loss on the test dataset using the best model.
    - best_model_state (dict): State dictionary of the best model.
    """
    best_epoch = -1
    best_validation_loss = float('inf')
    best_model_state = None

    def validate_model(net, criterion):
        logging.debug('Validating model...')
        tensor_x_validation, tensor_y_validation = dataset_manager.prepare_data(
            validation_dataset, label_key=label_key, label_encoder=label_encoder, device=device
        )
        net.eval()
        with torch.no_grad():
            outputs_validation = net(tensor_x_validation)
            validation_loss = criterion(outputs_validation, tensor_y_validation).item()
        return validation_loss

    def after_epoch(net, epoch):
        nonlocal best_epoch, best_validation_loss, best_model_state
        validation_loss = validate_model(net, nn.CrossEntropyLoss())
        logging.debug('Epoch %d validation loss: %.4f', epoch + 1, validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_model_state = net.state_dict()

    # Train and evaluate
    logging.debug('Starting training and validation process...')
    _ = train_and_evaluate_mlp(
        adata_train=train_dataset,
        adata_test=None,
        label_key=label_key,
        label_encoder=label_encoder,
        num_classes=num_classes,
        epoch_num=epoch_num,
        device=device,
        batch_size=batch_size,
        format_manager=dataset_manager,
        run_after_epoch=after_epoch
    )

    # Load the best model state
    logging.debug('Loading best model state from epoch %d', best_epoch + 1)
    net = models.Net(dataset_manager.get_feature_size(train_dataset), output_size=num_classes)
    net.to(device)
    net.load_state_dict(best_model_state)

    # Test evaluation
    logging.debug('Evaluating on test dataset...')
    tensor_x_test, tensor_y_test = dataset_manager.prepare_data(
        test_dataset, label_key=label_key, label_encoder=label_encoder, device=device
    )
    net.eval()
    with torch.no_grad():
        outputs_test = net(tensor_x_test)
        test_loss = nn.CrossEntropyLoss()(outputs_test, tensor_y_test).item()

    return best_epoch, test_loss, best_model_state

def get_subset_composition(adata, group_counts):

    # Get the indices of each group
    easy_indices = adata.obs.index[adata.obs['Annotation'] == 'Easy-to-learn'].tolist()
    ambiguous_indices = adata.obs.index[adata.obs['Annotation'] == 'Ambiguous'].tolist()
    hard_indices = adata.obs.index[adata.obs['Annotation'] == 'Hard-to-learn'].tolist()

    e = group_counts.get('Easy-to-learn', 0)
    a = group_counts.get('Ambiguous', 0)
    h = group_counts.get('Hard-to-learn', 0)

    train_easy_indices = random.sample(easy_indices, e) if e > 0 else []
    train_ambiguous_indices = random.sample(ambiguous_indices, a) if a > 0 else []
    train_hard_indices = random.sample(hard_indices, h) if h > 0 else []
    train_indices = train_easy_indices + train_ambiguous_indices + train_hard_indices

    # Ensure total train samples equal T
    if len(train_indices) != sum([e, a, h]):
        raise ValueError('Train size mismatch.')
    
    # Create training and testing datasets
    return adata[train_indices].copy(), train_indices

def run_single_subset_evaluation(train_indices, adata, dataset_manager, label_key, epoch_num, device, batch_size):
    """
    Given a set of training indices:
        - Remove them from 'all_indices' to define leftover
        - From leftover, pick val_size_sub (10%) and test_size_sub (20%)
        - Train a model, return test_loss
    """
    subset_size = len(train_indices)
    val_size_sub = int(0.1 * subset_size)
    test_size_sub = int(0.2 * subset_size)

    all_indices = np.arange(adata.n_obs)
    leftover_indices = np.setdiff1d(all_indices, train_indices)

    # Shuffle leftover to pick validation/test consistently
    np.random.shuffle(leftover_indices)

    if len(leftover_indices) < (val_size_sub + test_size_sub):
        raise ValueError(
            f"Not enough leftover indices to form val({val_size_sub}) and "
            f"test({test_size_sub}) after picking a subset of size {subset_size}."
        )

    val_indices_sub = leftover_indices[:val_size_sub]
    test_indices_sub = leftover_indices[val_size_sub: val_size_sub + test_size_sub]

    # Build actual AnnData subsets
    train_dataset = dataset_manager.subset(adata, train_indices)
    val_dataset   = dataset_manager.subset(adata, val_indices_sub)
    test_dataset  = dataset_manager.subset(adata, test_indices_sub)

    # Retrieve label encoder
    label_encoder = dataset_manager.getLabelEncoder(adata, label_key)

    # Train and evaluate
    _, test_loss, _ = train_validate_and_evaluate(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        dataset_manager=dataset_manager,
        label_key=label_key,
        label_encoder=label_encoder,
        num_classes=len(label_encoder.classes_),
        epoch_num=epoch_num,
        device=device,
        batch_size=batch_size
    )
    return test_loss

def gather_and_aggregate_results(
    dataset_name,
    results_dir="results",
    final_csv="best_compositions.csv"
):
    """
    Reads all results_*.json from `results_dir`, aggregates runs with the same 
    (Train_Size, Easy, Ambiguous, Hard) by computing the average test_loss, and
    then finds the best composition per Train_Size (lowest avg loss).

    Saves the best compositions to `final_csv`.
    """

    logging.info(f"Gathering results from directory: {results_dir}")
    all_rows = []

    # 1) Read each JSON result file
    for fn in os.listdir(results_dir):
        if (fn.startswith("results_") and fn.endswith(".json")):
            path = os.path.join(results_dir, fn)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                # Data has keys like:
                # {
                #   "row_id": <int>,
                #   "Train_Size": <int>,
                #   "Easy": <int>,
                #   "Ambiguous": <int>,
                #   "Hard": <int>,
                #   "Run": <int>,
                #   "Test_Indices": [...],
                #   "Train_Indices": [...],
                #   "Test_Loss": <float or null>
                # }
                all_rows.append(data)
            except Exception as ex:
                logging.warning(f"Failed to load {path}: {ex}")

    if not all_rows:
        logging.warning(f"No JSON files found in {results_dir}. Nothing to aggregate.")
        return None

    # 2) Convert to a DataFrame
    df = pd.DataFrame(all_rows)

    # Ensure the relevant columns exist
    required_cols = ["Train_Size", "Easy", "Ambiguous", "Hard", "Test_Loss", "Run", 
                        "Train_Indices", "Test_Indices"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.error(f"Missing columns in the results data: {missing}")
        return None

    # We only aggregate rows that have a valid test_loss
    valid_df = df.dropna(subset=["Test_Loss"]).copy()
    if valid_df.empty:
        logging.warning("No valid results with 'Test_Loss' found. All are NaN or None?")
        return None

    valid_df["Test_Loss"] = pd.to_numeric(valid_df["Test_Loss"], errors="coerce")

    # 3) Group by (Train_Size, Easy, Ambiguous, Hard) and compute average test_loss
    #    We'll also gather the run indices and train_indices for each group.
    agg_funcs = {
        "Test_Loss": ["mean", "count"],  # avg test loss, plus how many runs
        "Run": lambda x: list(x),
        "Train_Indices": lambda x: list(x),  # store a list-of-lists
        "Test_Indices": "first"  # typically all runs in the same group share the same test set
    }
    grouped = valid_df.groupby(["Train_Size", "Easy", "Ambiguous", "Hard"]).agg(agg_funcs)

    # Flatten multi-level columns from the groupby
    grouped.columns = [
        "_".join(col).rstrip("_") for col in grouped.columns.to_flat_index()
    ]
    # So we get columns like:
    #   Test_Loss_mean, Test_Loss_count, Run_<lambda>, Train_Indices_<lambda>, Test_Indices_first

    grouped = grouped.reset_index()

    grouped.rename(columns={
        "Test_Loss_mean": "Avg_Test_Loss",
        "Test_Loss_count": "Num_Runs",
        "Run_<lambda>": "All_Runs",
        "Train_Indices_<lambda>": "All_Train_Indices",
        "Test_Indices_first": "Test_Indices"
    }, inplace=True)

    # Now we have a DataFrame like:
    # Train_Size | Easy | Ambiguous | Hard | Avg_Test_Loss | Num_Runs | All_Runs | All_Train_Indices | Test_Indices

    # 4) For each Train_Size, find the row with the smallest Avg_Test_Loss
    idxmin_series = grouped.groupby("Train_Size")["Avg_Test_Loss"].idxmin()
    best_per_size = grouped.loc[idxmin_series].copy()

    # 5) Write out the best compositions for each train size
    best_per_size.sort_values("Train_Size", inplace=True)
    best_per_size.to_csv(final_csv, index=False)

    logging.info(f"Wrote best compositions to {final_csv}")

    return best_per_size

def comp_opt_subset_to_not(
    dataset_name,
    adata,
    label_key,
    group_counts,
    epoch_num_subset,
    epoch_num_full,
    batch_size,
    dataset_manager,
    repeats_per_size=5,
    model="mlp",
    random_seed=42
):
    """
    This function:
      1) Runs multiple passes (repeats_per_size) for both 'optimal' and 'random' subsets:
         - Each run re-generates a training subset (optimal vs. random) of the same size,
           then splits leftover into val (10%) and test (20%) of that subset size.
         - Test loss is recorded for each run.
         - The final 'optimal' and 'random' test losses are averaged across repeats_per_size runs.
      2) Performs a single pass with the 'full' dataset (70/10/20 split).
      3) Logs and writes all results to CSV with columns:
         ['train_indices', 'test_loss', 'type'].
         The 'type' field is one of 'optimal', 'random', or 'full'.
      4) Logs the average test loss of 'optimal' and 'random' runs,
         and the single test loss for 'full'.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # 1) REPEATED RUNS: OPTIMAL
    # ------------------------------------------------------------------
    optimal_losses = []
    optimal_run_details = []

    for i in range(repeats_per_size):
        _, optimal_train_indices = get_subset_composition(adata, group_counts)
        optimal_train_indices = np.array(optimal_train_indices)

        test_loss_opt = run_single_subset_evaluation(optimal_train_indices, adata, dataset_manager, label_key, epoch_num_subset, device, batch_size)
        optimal_losses.append(test_loss_opt)
        
        # Store for CSV
        optimal_run_details.append({
            "type": "optimal",
            "test_loss": test_loss_opt,
            "train_indices": optimal_train_indices.tolist(),
        })

        logging.info(f"[Optimal run {i+1}/{repeats_per_size}] test_loss = {test_loss_opt:.4f}")

    avg_optimal_loss = float(np.mean(optimal_losses))
    logging.info(f"Average test loss for 'optimal' (over {repeats_per_size} runs): {avg_optimal_loss:.4f}")

    # ------------------------------------------------------------------
    # 2) REPEATED RUNS: RANDOM
    # ------------------------------------------------------------------

    subset_size = len(optimal_run_details[0]["train_indices"])

    random_losses = []
    random_run_details = []

    for i in range(repeats_per_size):
        # Sample random training subset of the same size
        all_indices = np.arange(adata.n_obs)
        rand_train_indices = np.random.choice(all_indices, size=subset_size, replace=False)

        test_loss_random = run_single_subset_evaluation(rand_train_indices, adata, dataset_manager, label_key, epoch_num_subset, device, batch_size)
        random_losses.append(test_loss_random)

        # Store for CSV
        random_run_details.append({
            "type": "random",
            "test_loss": test_loss_random,
            "train_indices": rand_train_indices.tolist()
        })

        logging.info(f"[Random run {i+1}/{repeats_per_size}] test_loss = {test_loss_random:.4f}")

    avg_random_loss = float(np.mean(random_losses))
    logging.info(f"Average test loss for 'random' (over {repeats_per_size} runs): {avg_random_loss:.4f}")

    # ------------------------------------------------------------------
    # 3) SINGLE RUN: FULL DATASET (70/10/20 split)
    # ------------------------------------------------------------------
    n_obs = adata.n_obs
    val_size_full = int(0.1 * n_obs)
    test_size_full = int(0.2 * n_obs)

    all_indices_full = np.arange(n_obs)
    np.random.shuffle(all_indices_full)

    val_indices_full = all_indices_full[:val_size_full]
    test_indices_full = all_indices_full[val_size_full: val_size_full + test_size_full]
    train_indices_full = all_indices_full[val_size_full + test_size_full:]

    # Prepare the full subsets
    full_train_dataset = dataset_manager.subset(adata, train_indices_full)
    full_val_dataset   = dataset_manager.subset(adata, val_indices_full)
    full_test_dataset  = dataset_manager.subset(adata, test_indices_full)

    # One pass with the "full" approach
    label_encoder = dataset_manager.getLabelEncoder(adata, label_key)
    _, test_loss_full, _ = train_validate_and_evaluate(
        train_dataset=full_train_dataset,
        validation_dataset=full_val_dataset,
        test_dataset=full_test_dataset,
        dataset_manager=dataset_manager,
        label_key=label_key,
        label_encoder=label_encoder,
        num_classes=len(label_encoder.classes_),
        epoch_num=epoch_num_full,
        device=device,
        batch_size=batch_size
    )

    logging.info(f"[Full] test_loss = {test_loss_full:.4f}")

    # ------------------------------------------------------------------
    # 4) WRITE RESULTS TO CSV
    # ------------------------------------------------------------------
    csv_filename = f"{dataset_name}_{group_counts}_eval.csv"
    results = []

    # Add all 'optimal' runs
    results.extend(optimal_run_details)

    # Add all 'random' runs
    results.extend(random_run_details)

    # Add single 'full' run
    results.append({
        "type": "full",
        "test_loss": test_loss_full,
        "train_indices": train_indices_full.tolist()
    })

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["type",  "test_loss", "train_indices"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logging.info(f"Results written to {csv_filename}")

    # ------------------------------------------------------------------
    # 5) FINAL LOGGING
    # ------------------------------------------------------------------
    logging.info(
        "Summary of results: "
        f"Optimal test loss (avg of {repeats_per_size}) = {avg_optimal_loss:.4f}, "
        f"Random test loss (avg of {repeats_per_size}) = {avg_random_loss:.4f}, "
        f"Full test loss (single run) = {test_loss_full:.4f}"
    )