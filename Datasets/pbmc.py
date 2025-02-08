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
        self.adata_pbmc = sc.read_h5ad(FILE_PATH)
        self.n_pca_components = 100  # Reduced to 100 principal components
        self.adata_pbmc = self.preprocess_data()
        self.label_key = 'cell_type'

        # Store original features in uns
        self.adata_pbmc.uns['original_features'] = self.adata_pbmc.var_names.copy()
        
        # Parameters for pbmc (full)
        self.epoch_num_annot = 40
        self.epoch_num_composition = 20
        self.swap_probability = 0.1
        self.percentile = 90
        self.batch_size = 64
        self.manager = AnnDataManager()
        self.name = "pbmc"

        return self.adata_pbmc

    def preprocess_data(self):
        # Normalize total (skip if already normalized)
        sc.pp.normalize_total(self.adata_pbmc, target_sum=1e4)
        
        # Store normalized data in layers before any other processing
        self.adata_pbmc.layers['normalized'] = self.adata_pbmc.X.copy()
        
        # Handle missing values before scaling
        sc.pp.filter_genes(self.adata_pbmc, min_cells=1)  # Remove genes that are all-zero
        sc.pp.filter_cells(self.adata_pbmc, min_genes=1)  # Remove cells that are all-zero
        
        # Handle remaining NaN values by imputation
        if scipy.sparse.issparse(self.adata_pbmc.X):
            self.adata_pbmc.X = self.adata_pbmc.X.toarray()
        mask = np.isnan(self.adata_pbmc.X)
        self.adata_pbmc.X[mask] = 0  # Replace NaN with zeros
        
        # Scale data for PCA
        sc.pp.scale(self.adata_pbmc)
        
        # Perform PCA
        sc.tl.pca(self.adata_pbmc, n_comps=self.n_pca_components, svd_solver='arpack')
        
        # Create a new AnnData object with PCA components
        adata_pca = sc.AnnData(
            X=self.adata_pbmc.obsm['X_pca'],
            obs=self.adata_pbmc.obs,
            var=pd.DataFrame(
                index=[f'PC{i+1}' for i in range(self.n_pca_components)],
                columns=['pca_component']
            )
        )
        
        # Copy important attributes
        adata_pca.uns = self.adata_pbmc.uns
        adata_pca.obsm = self.adata_pbmc.obsm
        
        return adata_pca

    def filter_by_health(self, clear_sick=True, normalize_again=False):
        if HEALTH_COLUMN not in self.adata_pbmc.obs.columns:
            raise KeyError(f"{HEALTH_COLUMN} column not found in adata_full.obs.")

        if HEALTHY_LABEL not in self.adata_pbmc.obs[HEALTH_COLUMN].unique():
            raise ValueError(f"{HEALTHY_LABEL} label not found in {HEALTH_COLUMN} column.")

        logging.info(f"Unique values in {HEALTH_COLUMN} column: "
                     f"{self.adata_pbmc.obs[HEALTH_COLUMN].unique()})")

        if clear_sick:
            filter_condition = self.adata_pbmc.obs[HEALTH_COLUMN] == HEALTHY_LABEL 
            status = "healthy"
            self.name = "pbmc_healthy"
        else:
            filter_condition = self.adata_pbmc.obs[HEALTH_COLUMN] != HEALTHY_LABEL
            status = "sick"
            self.name = "pbmc_sick"

        # Apply filtering
        filtered_adata = self.adata_pbmc[filter_condition].copy()
        
        if filtered_adata.n_obs == 0:
            raise ValueError(f"No {status} cells found after filtering.")

        self.adata_pbmc = filtered_adata
        print(f"Filtered {status} cells using {HEALTH_COLUMN}.")

        if normalize_again:
            # Re-normalize using stored normalized layer
            self.adata_pbmc.X = self.adata_pbmc.layers['normalized']
            return self.preprocess_data()
        
        return self.adata_pbmc

    # Add method to access original features if needed
    def get_original_features(self):
        return self.adata_pbmc.uns.get('original_features', None)