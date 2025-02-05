
from dataset import Dataset
import scanpy as sc
import squidpy as sq

class Merfish(Dataset):
    def load_data(self):
        # Load data and save it as an instance attribute
        self.adata_merfish = sq.datasets.merfish()
        return self.adata_merfish

    def preprocess_data(self):
        sc.pp.normalize_per_cell(self.adata_merfish, counts_per_cell_after=1e4)
        sc.pp.log1p(self.adata_merfish)

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
        self.adata_merfish.obs['CellType'] = self.adata_merfish.obs['Cell_class'].map(cell_type_mapping).fillna(self.adata_merfish.obs['Cell_class'])
        return self.adata_merfish
