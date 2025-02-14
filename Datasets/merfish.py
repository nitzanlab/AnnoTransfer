
from Datasets.dataset import Dataset
from Managers.anndata_manager import AnnDataManager
import scanpy as sc
import logging
import squidpy as sq

class Merfish(Dataset):
    def load_data(self):
        self.adata = sq.datasets.merfish()
        self.adata = self.preprocess_data()

        # Parameters for merfish
        self.label_key = 'CellType'
        self.epoch_num_annot = 150
        self.epoch_num_composition = 30
        self.swap_probability = 0.1
        self.percentile = 90
        self.batch_size = 64
        self.manager = AnnDataManager()
        self.name = "merfish"

        return self.adata

    def preprocess_data(self):
        sc.pp.normalize_per_cell(self.adata, counts_per_cell_after=1e4)
        sc.pp.log1p(self.adata)

        logging.info("Existing cell types: ")
        logging.info(self.adata.obs['Cell_class'].value_counts())

        # Map clusters to cell types
        cell_type_mapping = {
            'OD Mature 2': 'OD Mature',
            'OD Immature 1': 'OD Immature',
            'Inhibitory': 'Inhibitory',
            'Excitatory': 'Excitatory',
            'Microglia': 'Microglia',
            'Astrocyte': 'Astrocyte',
            'Endothelial 2': 'Endothelial',
            'Endothelial 3': 'Endothelial',
            'Endothelial 1': 'Endothelial',
            'OD Mature 1': 'OD Mature',
            'OD Mature 4': 'OD Mature',
            'Pericytes': 'Pericytes',
            'OD Mature 3': 'OD Mature',
            'Ependymal': 'Ependymal',
            'OD Immature 2': 'OD Immature'
        }
        self.adata.obs['CellType'] = self.adata.obs['Cell_class'].map(cell_type_mapping).fillna(self.adata.obs['Cell_class'])
        logging.info("Transformed to: ")
        logging.info(self.adata.obs['Cell_class'].value_counts())

        return self.adata
