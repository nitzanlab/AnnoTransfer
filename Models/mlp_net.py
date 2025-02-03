import warnings

warnings.filterwarnings("ignore")
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
from Managers.anndata_manager import *

def one_hot_encode(labels, label_encoder):
    logging.debug('One-hot encoding labels...')
    values = np.array(labels)
    integer_encoded = label_encoder.transform(values)
    onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    logging.debug('One-hot encoding complete.')
    return onehot_encoded

def is_scipy_cs_sparse(matrix):
    logging.debug('Checking if matrix is scipy csr sparse...')
    result = sp.issparse(matrix) and matrix.getformat() == 'csr'
    logging.debug('Matrix is scipy csr sparse: %s', result)
    return result

def train_and_evaluate_mlp(
    adata_train,
    adata_test,
    label_key,
    label_encoder,
    num_classes,
    epoch_num,
    device,
    batch_size,
    format_manager,
    run_after_epoch=None
):
    """
    Trains and evaluates a neural network model on the provided training and testing data.

    Parameters:
    - adata_train (AnnData): Training dataset.
    - adata_test (AnnData): Testing dataset.
    - label_key (str): Key in adata.obs that contains the labels.
    - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
    - num_classes (int): Number of unique classes in the dataset.
    - epoch_num (int): Number of training epochs.
    - device (str or torch.device): Device to run the training on ('cpu' or 'cuda').
    - batch_size (int): Batch size for training.

    Returns:
    - test_loss (float): Loss on the test dataset.
    """
    # Initialize the neural network
    logging.debug('Initializing the neural network...')
    net = models.Net(adata_train.X.shape[1], output_size=num_classes)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Prepare training data
    logging.debug('Preparing training data...')
    tensor_x_train, tensor_y_train = format_manager.prepare_data(
        adata=adata_train, label_key=label_key, label_encoder=label_encoder, device=device
    )
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the network
    net.train()
    for epoch in range(epoch_num):
        logging.debug('Epoch %d/%d', epoch + 1, epoch_num)
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if run_after_epoch:
            run_after_epoch(net, epoch)

    if adata_test:
        # Prepare test data
        logging.debug('Preparing test data...')
        tensor_x_test, tensor_y_test = format_manager.prepare_data(
            adata=adata_test, label_key=label_key, label_encoder=label_encoder, device=device
        )

        # Evaluate on test set
        net.eval()
        with torch.no_grad():
            outputs = net(tensor_x_test)
            test_loss = criterion(outputs, tensor_y_test).item()

        return test_loss
    
    return None