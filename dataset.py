import torch
import numpy as np
import pandas as pd
import mutagenicity_utils
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms
from tqdm import tqdm
from rdkit.Chem import Fragments


def load_mutagenicity_dataset(dataset_root):
    transform = torch_geometric.transforms.Compose([
        mutagenicity_utils.MapNodeLabels(),
        mutagenicity_utils.MapEdgeLabels(),
        mutagenicity_utils.MapGraphClassLabel(),
        mutagenicity_utils.NumUndirectedEdges()
    ])
    return TUDataset(root=dataset_root, name='Mutagenicity', transform=transform)

def get_frag_query_map(queryset):
    """
    Create a mapping from fragment names to their counts in the queryset.
    This function groups the queryset by 'frag_name' and aggregates the 'count' values.
    Input:
        queryset (pd.DataFrame): DataFrame containing 'frag_name' and 'count' columns.
    Output:
        dict: A dictionary mapping fragment names to their corresponding counts.
    """
    frag_query_map = {
        name: group['count'].tolist()
        for name, group in queryset.groupby('frag_name')
    }
    return frag_query_map


def load_mutagenicity_query_answer_dataset(dataset_root, queryset_root, device, train_ratio=None, seed=None):
    print('Loading and processing Mutagenicity dataset for training...')

    # Create list of onehot query-answer vectors
    raw_dataset = load_mutagenicity_dataset(dataset_root)
    queryset = pd.read_csv(queryset_root)
    frag_query_map = get_frag_query_map(queryset)

    x = torch.empty((len(raw_dataset), len(queryset)))  # (n_datapoints, n_queries)
    for i, data in tqdm(enumerate(raw_dataset), total=len(raw_dataset)):
        mol = mutagenicity_utils.raw_to_rdkit(data)
        qry_ans_vec = mutagenicity_utils.mol_to_query_answers(mol, frag_query_map, device)  # (1, n_queries)
        x[i] = qry_ans_vec
    y = raw_dataset.y.unsqueeze(1)#.to(device)  # (n_datapoints, 1) 
    raw_data_ids = torch.arange(0, len(raw_dataset)).unsqueeze(1)  # (n_datapoints, 1)
    dataset = torch.utils.data.TensorDataset(x, y, raw_data_ids)

    if train_ratio == None:
        print("Loading complete without splitting into train/test.")
        return dataset
    else:
        print('Splitting into train and test sets...')
        # Split dataset into train and test
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size

        if seed != None:
            generator=torch.Generator().manual_seed(seed)  # Seed for random split
        else:
            generator=None  # No seed
        trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

        print("Loading complete.")
        return trainset, testset