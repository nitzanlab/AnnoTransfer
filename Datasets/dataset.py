from abc import ABC, abstractmethod
from anndata import AnnData
import scanpy as sc
import os

class Dataset(ABC):
    @abstractmethod
    def load_data(self) -> AnnData:
        pass

    @abstractmethod
    def preprocess_data(self) -> AnnData:
        pass

    def get_annotated_dataset(self) -> AnnData:
        # The name attribute should be defined in the class or passed to the method
        if not hasattr(self, 'name'):
            raise AttributeError("Dataset name not defined. Please set self.name in the implementing class.")

        annotated_path = f"{self.name}_annotated.h5ad"
        if not os.path.exists(annotated_path):
            raise FileNotFoundError(
                f"{annotated_path} not found - please run the annotation on the dataset first"
            )
        
        # Don't store as instance variable unless needed
        return sc.read(annotated_path)
