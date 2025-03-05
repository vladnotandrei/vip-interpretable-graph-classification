import os
import torch
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import mutagenicity_utils
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms
from tqdm import tqdm
    
    
# def load_mnist(root):
#     transform = transforms.Compose([transforms.ToTensor(),  
#                                     transforms.Lambda(lambda x: torch.where(x < 0.5, -1., 1.))])
#     trainset = datasets.MNIST(root, train=True, transform=transform, download=True)
#     testset = datasets.MNIST(root, train=False, transform=transform, download=True)
#     return trainset, testset


def load_news(root):
    # read data from pickle file
    with open(f"{root}/cleaned_categories10.pkl", "rb") as f:
        data = pickle.load(f)
        x, y = data["x"].toarray(), data["y"]
        label_ids, vocab = data["label_ids"], data["vocab"]

    # binarize by thresholding 0
    x = np.where((x > 0), np.ones(x.shape), -np.ones(x.shape))
    x = np.float32(x)

    # split into sub-datasets
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train, val, test = torch.utils.data.random_split(
        dataset,
        [
            round(0.8 * len(dataset)),
            round(0.1 * len(dataset)),
            len(dataset) - round(0.8 * len(dataset)) - round(0.1 * len(dataset)),
        ],
        torch.Generator().manual_seed(42),  # Use same seed to split data
    )
    return train, val, test, vocab, list(label_ids)


class CUB200(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(
        self,
        root,
        image_dir='CUB_200_2011',
        split='train',
        transform=None,
):
        
        self.root = root
        self.image_dir = os.path.join(self.root, 'CUB', image_dir)
        self.transform = transform

        ## Image
        pkl_file_path = os.path.join(self.root, 'CUB', f'{split}class_level_all_features.pkl')
        self.data = []
        with open(pkl_file_path, "rb") as f:
            self.data.extend(pickle.load(f))
            
        ## Classes
        self.classes = pd.read_csv(os.path.join(self.image_dir, 'classes.txt'), header=None).iloc[:, 0].values


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _dict = self.data[idx]

        # image
        img_path = _dict['img_path']
        _idx = img_path.split("/").index("CUB_200_2011")
        img_path = os.path.join(self.root, 'CUB/CUB_200_2011', *img_path.split("/")[_idx + 1 :])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # class label
        class_label = _dict["class_label"]
        return img, class_label


# def load_cub(root):    
#     transform = transforms.Compose(
#         [
#             transforms.CenterCrop(299),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
#         ]
#     )
#     trainset = CUB200(root, image_dir='CUB_200_2011', split='train', transform=transform)
#     testset = CUB200(root, image_dir='CUB_200_2011', split='test', transform=transform)
#     valset = CUB200(root, image_dir='CUB_200_2011', split='val', transform=transform)
#     return trainset, valset, testset
    
# def load_cifar10(root):
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     trainset = datasets.CIFAR10(
#         root=root, train=True, download=True, transform=transform_train)
#     testset = datasets.CIFAR10(
#         root=root, train=False, download=True, transform=transform_test)
#     return trainset, testset


def load_mutagenicity_dataset(dataset_root):
    transform = torch_geometric.transforms.Compose([
        mutagenicity_utils.MapNodeLabels(),
        mutagenicity_utils.MapEdgeLabels(),
        mutagenicity_utils.MapGraphClassLabel(),
        mutagenicity_utils.NumUndirectedEdges()
    ])
    return TUDataset(root=dataset_root, name='Mutagenicity', transform=transform)


def load_mutagenicity_queryset(dataset_root, queryset_root, train_ratio=None):
    print('Loading and processing Mutagenicity dataset for training...')

    raw_dataset = load_mutagenicity_dataset(dataset_root)
    
    raw_queryset = pd.read_csv(queryset_root)
    frag_func_names = raw_queryset['frag_func_name'].to_list()
    count_list = [eval(n) for n in raw_queryset['count_list'].to_list()]

    # Create list of onehot query-answer vectors
    x, y = [], []  # labels, query vector
    for data in tqdm(raw_dataset):
        G = mutagenicity_utils.raw_to_nx(data)
        mol = mutagenicity_utils.nx_to_rdkit(G)
        onehot = mutagenicity_utils.fragment_occurence_counts_onehot(mol, frag_func_names, count_list)
        x.append(onehot)
        y.append(data.y.item())  # label is in singleton list in tensor, extract value from tensor and list
    x = np.array(x)
    y = np.array(y)
    raw_data_ids = np.array(range(0, len(raw_dataset)))

    # Create pytorch dataset
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), 
                                             torch.from_numpy(y), 
                                             torch.from_numpy(raw_data_ids))

    # TODO: Any more processing to do before training?

    if train_ratio == None:
        print("Loading complete.")
        return dataset
    else:
        print('Splitting into train and test sets...')
        # Split dataset into train and test
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # TODO: See how TUDataset recommends doing the train/test split for Mutagenicity.
        print("Loading complete.")
        return trainset, testset