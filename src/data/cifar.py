'''Loads the CIFAR10 and the CIFAR100 datasets'''

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Load CIFAR10 dataset
class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.dataset = datasets.CIFAR10(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

# Load CIFAR100 dataset
class CIFAR100(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.dataset = datasets.CIFAR100(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.classes = [] #enumerates all the classes in the CIFAR100 dataset
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    