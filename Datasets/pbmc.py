
from Datasets.dataset import Dataset
from Managers.anndata_manager import AnnDataManager
import scanpy as sc
import squidpy as sq
import logging
import os

FILE_PATH = os.path.join(os.environ['PROJECT_DIR'], "Datasets", "pbmc_cvid.h5ad")
HEALTHY_LABEL = 'normal'
HEALTH_COLUMN = 'disease'

class PBMC(Dataset):
    def load_data(self):
        # Load data and save it as an instance attribute
        self.adata_pbmc = sc.read_h5ad(FILE_PATH)
        self.adata_pbmc = self.preprocess_data()
        self.label_key = 'cell_type'

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
        # Normalize and log-transform the data
        sc.pp.normalize_total(self.adata_pbmc, target_sum=1e4)
        sc.pp.log1p(self.adata_pbmc)
        return self.adata_pbmc

    def filter_by_health(self, clear_sick=True, normalize_again=False):
        if HEALTH_COLUMN not in self.adata_pbmc.obs.columns:
            raise KeyError("HEALTH_COLUMN column not found in adata_full.obs.")

        if HEALTHY_LABEL not in self.adata_pbmc.obs[HEALTH_COLUMN].unique():
            raise ValueError("HEALTHY_LABEL label not found in HEALTH_COLUMN column.")

        # print all unique values in HEALTH_COLUMN as a log
        logging.info(f"Unique values in {HEALTH_COLUMN} column: {self.adata_pbmc.obs[HEALTH_COLUMN].unique()})")

        if clear_sick:
            filter_condition = self.adata_pbmc.obs[HEALTH_COLUMN] == HEALTHY_LABEL 
            status = "healthy"
            # parameters for pbmc_healthy
            self.epoch_num_annot = 50
            self.epoch_num_composition = 25
            self.swap_probability = 0.1
            self.percentile = 90
            self.batch_size = 64
            self.name = "pbmc_healthy"

        else:
            filter_condition = self.adata_pbmc.obs[HEALTH_COLUMN] != HEALTHY_LABEL
            status = "sick"
            # parameters for pbmc_sick
            self.epoch_num_annot = 50
            self.epoch_num_composition = 25
            self.swap_probability = 0.1
            self.percentile = 90
            self.batch_size = 64
            self.name = "pbmc_sick"

        filtered_adata = self.adata_pbmc[filter_condition].copy()
        
        if filtered_adata.n_obs == 0:
            raise ValueError(f"No {HEALTHY_LABEL if clear_sick else 'sick'} cells found after filtering.")

        self.adata_pbmc = filtered_adata

        print(f"Filtered {status} cells using HEALTH_COLUMN.")

        if normalize_again:
            return self.preprocess_data()
        
        return self.adata_pbmc