import numpy as np
import torch
import scipy.sparse as sp
import logging
from sklearn.preprocessing import LabelEncoder

class AnnDataManager:
    """
    A simple manager class providing functionalities for handling AnnData objects:
    - Checking whether a matrix is a scipy CSR sparse matrix
    - One-hot encoding labels
    - Preparing data for PyTorch
    - Getting a fitted LabelEncoder for AnnData
    - Creating a subset of an AnnData
    - Logging general information about the dataset
    """

    def __init__(self):
        # No special initialization required for now
        pass

    def is_scipy_cs_sparse(self, matrix):
        """
        Checks if the given matrix is a scipy CSR sparse matrix.

        Parameters:
        - matrix: Input matrix to check.

        Returns:
        - bool: True if the matrix is a scipy CSR sparse matrix, False otherwise.
        """
        logging.debug('Checking if matrix is scipy csr sparse...')
        result = sp.issparse(matrix) and matrix.getformat() == 'csr'
        logging.debug('Matrix is scipy csr sparse: %s', result)
        return result

    def one_hot_encode(self, labels, label_encoder):
        """
        Encodes categorical labels into a one-hot format using a fitted LabelEncoder.

        Parameters:
        - labels: List or array of labels to encode.
        - label_encoder (LabelEncoder): A fitted LabelEncoder instance.

        Returns:
        - np.ndarray: One-hot encoded labels.
        """
        logging.debug('One-hot encoding labels...')
        values = np.array(labels)
        integer_encoded = label_encoder.transform(values)
        onehot_encoded = np.eye(len(label_encoder.classes_))[integer_encoded]
        logging.debug('One-hot encoding complete.')
        return onehot_encoded

    def prepare_data(self, adata, label_key, label_encoder, device):
        """
        Prepares input data and labels from an AnnData object for use in PyTorch.

        Parameters:
        - adata (AnnData): Dataset to process.
        - label_key (str): Key in adata.obs containing the labels.
        - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
        - device (str or torch.device): The device to transfer tensors to.

        Returns:
        - tensor_x (torch.Tensor): Features tensor.
        - tensor_y (torch.Tensor): Labels tensor.
        """
        logging.debug('Preparing data...')
        if self.is_scipy_cs_sparse(adata.X):
            x_data = adata.X.toarray()
        else:
            x_data = np.array(adata.X)
        tensor_x = torch.Tensor(x_data).to(device)
        one_hot_labels = self.one_hot_encode(adata.obs[label_key], label_encoder=label_encoder)
        tensor_y = torch.LongTensor(np.argmax(one_hot_labels, axis=1)).to(device)
        logging.debug('Data preparation complete.')
        return tensor_x, tensor_y

    def getLabelEncoder(self, adata, label_key):
        """
        Returns a fitted LabelEncoder instance for the given dataset and label key.

        Parameters:
        - adata (AnnData): Dataset to encode labels for.
        - label_key (str): Key in adata.obs containing the labels.

        Returns:
        - LabelEncoder: Fitted LabelEncoder instance.
        """
        label_encoder = LabelEncoder()
        label_encoder.fit(adata.obs[label_key])
        return label_encoder

    def subset(self, adata, indices):
        """
        Creates a subset of the given AnnData object based on specified indices.

        Parameters:
        - adata (AnnData): The original dataset to subset.
        - indices (array-like): Indices of observations to include in the subset.

        Returns:
        - AnnData: A new AnnData object containing only the specified observations.
        """
        logging.debug(f'Creating subset with {len(indices)} indices.')
        subset_adata = adata[indices].copy()
        logging.debug('Subset creation complete.')
        return subset_adata

    def general_info(self, adata):
        """
        Prints general information about the AnnData object via logging.

        Parameters:
        - adata (AnnData): Dataset to print information about.
        """
        logging.info('General information about the dataset:')
        logging.info(f'Number of cells: {adata.n_obs}')
        logging.info(f'Number of features: {adata.n_vars}')

    def get_feature_size(self, adata):
        return adata.n_vars