import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torch.utils.data import Dataset

from data.cifar import CIFAR10, CIFAR100


class TransformedSubset(Dataset):
    """
    A dataset that wraps a subset and applies a given transform.
    """
    def __init__(self, subset, transform=None):
        """
        Args:
            subset (Subset): A Subset object.
            transform (callable, optional): A function/transform that takes in 
                an element from the subset and returns a transformed version.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

    def __len__(self):
        return len(self.subset)

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


def online_transform(dataset, transform = None):
    '''Transforms the dataset using the transforms and returns the dataset.
    Args:
        dataset (Dataset): Dataset to transform.
        transform (torchvision.transforms): Transform to apply. Default: None.
    Returns:
        Dataset: Transformed dataset.
    '''
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = TransformedSubset(dataset, transform) # if the dataset is a subset, wrap it in TransformedSubset
    else:
        dataset.transform = transform # if the dataset is not a subset, apply the transform to the dataset
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
        n_classes = 10
    elif dataset_name == 'cifar100':
        dataset['train'] = CIFAR100(root='./dataset', train=True, transform = transform, download=True)
        dataset['test'] = CIFAR100(root='./dataset', train=False, transform = transform, download=True)
        n_classes = 100
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
    
    
    return dataset, n_classes


def split(dataset, split_ratio = 0.15, val_size = 500, shuffle = True, seed = 2):
    '''Splits the dataset into train and validation sets.
    Args:
        dataset (Dataset): Dataset to split.
        split_ratio (float): Ratio of the split. Default: 0.15.
        val_size (int): Size of the validation set. Default: 500.
        shuffle (bool): Whether to shuffle the dataset before splitting. Default: True.
        seed (int): Random seed. Default: 2.
    Returns:
        Dataset: Train dataset.
        Dataset: Validation dataset.
    '''
    if isinstance(dataset, dict):
        raise TypeError # if the dataset is a dictionary, take the train dataset for the splitting
    dataset_size = len(dataset)
    if split_ratio < 1:
        split1_size = int(split_ratio * dataset_size)
    else:
        split1_size = int(split_ratio)
    split2_size = dataset_size - split1_size - val_size
    split1, split2, split3 = random_split(dataset, [split1_size, val_size, split2_size], generator=torch.Generator().manual_seed(seed))
    return split1, split2, split3


def get_dataloader(dataset, batch_size = 128, shuffle = True, num_workers = 0, pin_memory = True):
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
    if isinstance(dataset, dict):
        dl = {}
        for key in dataset:
            dl[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    else:
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dl


def prepare_data(args, val_size = 500, retain_train_data = False, online = True):
    '''Prepares the data for training.
    Args:
        args (argparse.Namespace): Arguments.
        val_size (int): Size of the validation set. Default: 500.
        retain_train_data (bool): Whether to retain the training data. Default: False.
    Returns:
        dict: Dictionary of dataloaders.
        int: Number of classes.
    '''
    data, n_classes = fetch_dataset(args.data_set)
    data['labeled'], data['val'], data['unlabeled'] = split(data['train'], split_ratio = float(args.subset_size), val_size = val_size, shuffle = True, seed = args.seed_split_seed)
    if not retain_train_data:
        del data['train']
    if online:
        return data, n_classes
    else:
        dataloaders = get_dataloader(data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
        return dataloaders, n_classes
    
    
def prepare_online(data, transform):
    '''Prepares the data for online training.
    Args:
        data (dict): Dictionary of datasets.
        transform (torchvision.transforms): Transform to apply.
    Returns:
        dict: Dictionary of dataloaders.
    '''
    transformed_dataset = {}  
    if isinstance(transform, dict):
        for keys in data:
            transformed_dataset[keys] = online_transform(data[keys], transform[keys])
    else:
        for keys in data:
            transformed_dataset[keys] = online_transform(data[keys], transform)
    dataloaders = get_dataloader(transformed_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    return dataloaders