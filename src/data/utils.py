import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from cifar import CIFAR10, CIFAR100

def count_distribution(dataset):
    """Count the distribution of the dataset.

    Args:
        dataset (Dataset): Dataset to count.

    Returns:
        dict: Distribution of the dataset.
    """
    distribution = {}
    for _, label in dataset:
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def prepare_online(dataset, transform = None):
    '''Transforms the dataset using the transforms and returns the dataset.
    Args:
        dataset (Dataset): Dataset to transform.
        transform (torchvision.transforms): Transform to apply. Default: None.
    Returns:
        Dataset: Transformed dataset.
    '''
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = torch.utils.data.ConcatDataset([dataset])
    dataset.transform = transform
    return dataset


class NormalizeTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x *= 2 #get it in 0 to 2 range
        x -= 1 #get it in -1 to 1 range
        return x
    

def fetch_dataset(dataset_name):
    '''Fetches the dataset.
    Args:
        dataset_name (str): Name of the dataset. Currently supported: cifar10, cifar100, svhn, mnist, fashionmnist, imagenet.
    Returns:
        Dataset: Dataset.
    '''
    dataset = {}
    transform = transforms.Compose([transforms.ToTensor(), NormalizeTransform()]) # base transform
    if dataset_name == 'cifar10':
        dataset['train'] = CIFAR10(root='./dataset', train=True, transform = transform, download=True)
        dataset['test'] = CIFAR10(root='./dataset', train=False, transform = transform, download=True)
    elif dataset_name == 'cifar100':
        dataset['train'] = CIFAR100(root='./dataset', train=True, transform = transform, download=True)
        dataset['test'] = CIFAR100(root='./dataset', train=False, transform = transform, download=True)
    elif dataset_name == 'svhn':
        raise NotImplementedError
    elif dataset_name == 'mnist':
        raise NotImplementedError
    elif dataset_name == 'fashionmnist':
        raise NotImplementedError
    elif dataset_name == 'imagenet':
        raise NotImplementedError
    else:
        raise ValueError('Dataset not found.')
    
    
    return dataset


def split(dataset, split_ratio = 0.15, shuffle = True, seed = 2):
    '''Splits the dataset into train and validation sets.
    Args:
        dataset (Dataset): Dataset to split.
        split_ratio (float): Ratio of the split. Default: 0.15.
        shuffle (bool): Whether to shuffle the dataset before splitting. Default: True.
        seed (int): Random seed. Default: 2.
    Returns:
        Dataset: Train dataset.
        Dataset: Validation dataset.
    '''
    if isinstance(dataset, dict):
        raise TypeError # if the dataset is a dictionary, take the train dataset for the splitting
    dataset_size = len(dataset)
    split1_size = int(split_ratio * dataset_size)
    split2_size = dataset_size - split1_size
    split1, split2 = random_split(dataset, [split1_size, split2_size], generator=torch.Generator().manual_seed(seed))
    return split1, split2


def get_dataloader(dataset, batch_size = 128, shuffle = True, num_workers = 4, pin_memory = True):
    '''Returns the dataloader for the dataset.
    Args:
        dataset (Dataset): Dataset to load.
        batch_size (int): Batch size. Default: 128.
        shuffle (bool): Whether to shuffle the dataset. Default: True.
        num_workers (int): Number of workers. Default: 4.
        pin_memory (bool): Whether to pin memory. Default: True.
    Returns:
        DataLoader: Dataloader for the dataset.
    '''
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)