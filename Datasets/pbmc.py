from Datasets.dataset import Dataset
from Managers.anndata_manager import AnnDataManager
import scanpy as sc
import squidpy as sq
import logging
import os
import pandas as pd
import numpy as np
import scipy

FILE_PATH = os.path.join(os.environ['PROJECT_DIR'], "Datasets", "pbmc_cvid.h5ad")
HEALTHY_LABEL = 'normal'
HEALTH_COLUMN = 'disease'

class PBMC(Dataset):
    def load_data(self):
        # Load data and save it as an instance attribute
        self.adata = sc.read_h5ad(FILE_PATH)
        self.n_pca_components = 100  # Reduced to 100 principal components
        self.adata = self.preprocess_data()
        self.label_key = 'cell_type'

        # Store original features in uns
        self.adata.uns['original_features'] = self.adata.var_names.copy()
        
        # Parameters for pbmc (full)
        self.epoch_num_annot = 40
        self.epoch_num_composition = 20
        self.swap_probability = 0.1
        self.percentile = 90
        self.batch_size = 64
        self.manager = AnnDataManager()
        self.name = "pbmc"

        return self.adata

    def preprocess_data(self):
        # Normalize total counts
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        
        # Store normalized data in layers before any other processing
        self.adata.layers['normalized'] = self.adata.X.copy()
        
        # Handle missing values before scaling
        sc.pp.filter_genes(self.adata, min_cells=1)
        sc.pp.filter_cells(self.adata, min_genes=1)
        
        # Handle remaining NaN values by imputation
        if scipy.sparse.issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()
        mask = np.isnan(self.adata.X)
        self.adata.X[mask] = 0
        
        # Scale data for PCA
        sc.pp.scale(self.adata)
        
        # Perform PCA
        sc.tl.pca(
            self.adata, 
            n_comps=self.n_pca_components, 
            svd_solver='randomized'
        )
        
        # Create a new AnnData object with PCA components
        adata_pca = sc.AnnData(
            X=self.adata.obsm['X_pca'],
            obs=self.adata.obs,
            var=pd.DataFrame(
                index=[f'PC{i+1}' for i in range(self.n_pca_components)],
                columns=['pca_component']
            )
        )
        
        # Copy important attributes
        adata_pca.uns = self.adata.uns
        adata_pca.obsm = self.adata.obsm
        
        return adata_pca

    def filter_by_health(self, clear_sick=True, normalize_again=False):
        if HEALTH_COLUMN not in self.adata.obs.columns:
            raise KeyError(f"{HEALTH_COLUMN} column not found in adata_full.obs.")

        if HEALTHY_LABEL not in self.adata.obs[HEALTH_COLUMN].unique():
            raise ValueError(f"{HEALTHY_LABEL} label not found in {HEALTH_COLUMN} column.")

        logging.info(f"Unique values in {HEALTH_COLUMN} column: "
                     f"{self.adata.obs[HEALTH_COLUMN].unique()})")

        if clear_sick:
            filter_condition = self.adata.obs[HEALTH_COLUMN] == HEALTHY_LABEL 
            status = "healthy"
            self.name = "pbmc_healthy"
        else:
            filter_condition = self.adata.obs[HEALTH_COLUMN] != HEALTHY_LABEL
            status = "sick"
            self.name = "pbmc_sick"

        # Apply filtering
        filtered_adata = self.adata[filter_condition].copy()
        
        if filtered_adata.n_obs == 0:
            raise ValueError(f"No {status} cells found after filtering.")

        self.adata = filtered_adata
        print(f"Filtered {status} cells using {HEALTH_COLUMN}.")

        if normalize_again:
            # Re-normalize using stored normalized layer
            self.adata.X = self.adata.layers['normalized']
            return self.preprocess_data()
        
        return self.adata

    # Add method to access original features if needed
    def get_original_features(self):
        return self.adata.uns.get('original_features', None)