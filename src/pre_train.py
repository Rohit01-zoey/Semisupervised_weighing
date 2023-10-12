'''We pre-train the teacher and student models on the labeled data using the following snippet'''
import torch
import os
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
from data import utils

from torch.utils.data import Dataset

# data = utils.fetch_dataset('cifar100')

# f1, f2 = random_split(data['train'], [0.15, 0.85])

# print(type(f1))

# sd = TransformedSubset(f1)

# print(type(sd))

# sd.transform = transforms.Compose([transforms.CenterCrop(10)])

# print(type(sd))
# print(sd.transform)

# dl = utils.get_dataloader(sd, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

# for (x, y) in dl:
#     print(y)
#     break


import torchvision
import matplotlib.pyplot as plt

# Dataset without any transformations
dataset = torchvision.datasets.CIFAR100(root='./dataset', train=True, transform=None, download=True)

# Fetch a sample from the dataset
img, label = dataset[0]

# Display the image
plt.imshow(img)
plt.title(f"Original Image - Label: {label}")
plt.savefig('original.png')
plt.show()

# Now apply transformations
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=1), # Always flip for the sake of demonstration
    torchvision.transforms.ToTensor(),
])
transformed_dataset = utils.TransformedSubset(dataset, transform=transform)

# Fetch and display the transformed image
transformed_img, transformed_label = transformed_dataset[0]
plt.imshow(transformed_img.permute(1, 2, 0))
plt.title(f"Transformed Image - Label: {transformed_label}")
plt.savefig('transformed.png')
plt.show()
