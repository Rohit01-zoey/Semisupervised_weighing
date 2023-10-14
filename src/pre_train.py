'''We pre-train the teacher and student models on the labeled data using the following snippet'''
import torch
import os
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
from data import utils as data_utils
from network import set_seed, setup_optim, train, eval
from models import resnetV2, utils as model_utils

from torch.utils.data import Dataset


import torchvision
import matplotlib.pyplot as plt
import argparse

# # Dataset without any transformations
# dataset = torchvision.datasets.CIFAR100(root='./dataset', train=True, transform=None, download=True)

# # Fetch a sample from the dataset
# img, label = dataset[0]

# # Display the image
# plt.imshow(img)
# plt.title(f"Original Image - Label: {label}")
# plt.savefig('original.png')
# plt.show()

# # Now apply transformations
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(p=1), # Always flip for the sake of demonstration
#     torchvision.transforms.ToTensor(),
# ])
# transformed_dataset = utils.TransformedSubset(dataset, transform=transform)

# # Fetch and display the transformed image
# transformed_img, transformed_label = transformed_dataset[0]
# plt.imshow(transformed_img.permute(1, 2, 0))
# plt.title(f"Transformed Image - Label: {transformed_label}")
# plt.savefig('transformed.png')
# plt.show()

def main(args):
    global CUDA
    CUDA = "cuda:"+args.cuda # which gpu to use
    set_seed(args.seed) # set seed for reproducibility
    
    # fetching the datasets and create the splits for labeled, validation and unlabeled data
    datasets, n_classes = data_utils.prepare_data(args, val_size = 500, retain_train_data = False, online = True)
    
    model = model_utils.get_model(args.model, num_classes = n_classes, seed = args.seed_init)
    model.to(CUDA)
    
    schedule = setup_optim(model, args)
    
    criteria = torch.nn.CrossEntropyLoss()
    
    online_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)) ])
    
    for epoch in range(args.epoch):
        model.train()
        dataloaders = data_utils.prepare_online(datasets, online_transform)
        train_loss = train(model, dataloaders['labeled'], schedule['optimizer'], criteria, CUDA)
        schedule['lr'].step()
        
        test_loss, test_acc = eval(model, dataloaders['test'], criteria, CUDA)
        print(f"Epoch: {epoch+1}/{args.epoch} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--subset_size', '-s', type=str, help="how much labelled dataset to use, for e.g., 10 percent is 0.1")
    # parser.add_argument('--mcd_loss', )
    parser.add_argument('--seed_split_seed', '-sss', type=int, default=2)
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--data_set', '-d', type=str,default="cifar100")
    parser.add_argument('--seed', '-se', type=int,default=2)
    parser.add_argument('--seed_init', '-si', type=int,default=2)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--cuda', '-c', type=str, help='which gpu to use',default='0')
    parser.add_argument('--checkpoint', '-ch', type=bool, default=False)
    #parser.add_argument('--teacher', '-t', type=int, default=1)
    args = parser.parse_args()
    
    main(args)