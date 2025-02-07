
from Datasets.dataset import Dataset
import scanpy as sc
import squidpy as sq
import logging
FILE_PATH = "$PROJECT_DIR/Datasets/pbmc_cvid.h5ad"
HEALTHY_LABEL = 'normal'
HEALTH_COLUMN = 'disease'

class PBMC(Dataset):
    def load_data(self):
        # Load data and save it as an instance attribute
        self.adata_pbmc = sc.read_h5ad(FILE_PATH)
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

        filter_condition = self.adata_pbmc.obs[HEALTH_COLUMN] == HEALTHY_LABEL if clear_sick else self.adata_pbmc.obs[HEALTH_COLUMN] != HEALTHY_LABEL
        filtered_adata = self.adata_pbmc[filter_condition].copy()
        
        if filtered_adata.n_obs == 0:
            raise ValueError(f"No {HEALTHY_LABEL if clear_sick else 'sick'} cells found after filtering.")

        self.adata_pbmc = filtered_adata

        status = "healthy" if clear_sick else "sick"
        print(f"Filtered {status} cells using HEALTH_COLUMN.")

        if normalize_again:
            return self.preprocess_data()
        
        return self.adata_pbmc
