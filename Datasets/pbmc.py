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
        # 1. Normalize and store original data
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Store original normalized data in a NEW layer
        self.adata.layers['original_normalized'] = self.adata.X.copy()
        
        # 2. Filter and scale
        sc.pp.filter_genes(self.adata, min_cells=1)
        sc.pp.filter_cells(self.adata, min_genes=1)
        sc.pp.scale(self.adata)
        
        # 3. Perform PCA
        sc.tl.pca(self.adata, n_comps=100, svd_solver='randomized')
        
        # 4. Store original features BEFORE modifying the object
        original_features = self.adata.var_names.copy()
        
        # 5. Replace X with PCA components while preserving original structure
        # ---------------------------------------------------------------------
        # Create dimension-aligned PCA matrix
        pca_matrix = self.adata.obsm['X_pca']
        
        # Create new var for PCA components
        new_var = pd.DataFrame(
            index=[f'PC{i+1}' for i in range(pca_matrix.shape[1])],
            data={
                'variance_ratio': self.adata.uns['pca']['variance_ratio'],
                'loadings': list(self.adata.varm['PCs'].T)  # Gene loadings per PC
            }
        )
        
        # 6. Rebuild AnnData IN PLACE
        # ---------------------------
        # Preserve critical metadata
        original_obs = self.adata.obs.copy()
        original_uns = self.adata.uns.copy()
        original_obsm = self.adata.obsm.copy()
        
        # Create new AnnData object with PCA dimensions
        self.adata = sc.AnnData(
            X=pca_matrix,
            obs=original_obs,
            var=new_var,
            uns=original_uns,
            obsm=original_obsm
        )
        
        # 7. Store original features for reference
        self.adata.uns['original_features'] = original_features
        
        return self.adata

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