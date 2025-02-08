import sys
from Datasets.merfish import Merfish
from Datasets.pbmc import PBMC

def get_dataset(dataset_name):
    if dataset_name == 'merfish':
        dataset = Merfish()
        dataset.load_data()
        dataset.preprocess_data()
    elif dataset_name == 'pbmc':
        dataset = PBMC()
        dataset.load_data()
        dataset.preprocess_data()
    elif dataset_name == 'pbmc_healthy':
        dataset = PBMC()
        dataset.load_data()
        dataset.preprocess_data()
        dataset.filter_by_health(clear_sick=True)
    elif dataset_name == 'pbmc_sick':
        dataset = PBMC()
        dataset.load_data()
        dataset.preprocess_data()
        dataset.filter_by_health(clear_sick=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return dataset

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python factory.py <dataset_name>")
        sys.exit(1)
        
    dataset_name = sys.argv[1]
    dataset = get_dataset(dataset_name)