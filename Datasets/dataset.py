from abc import ABC, abstractmethod
from anndata import AnnData

class Dataset(ABC):
    @abstractmethod
    def load_data(self) -> AnnData:
        pass

    @abstractmethod
    def preprocess_data(self) -> AnnData:
        pass
