from metrics.acc import accuracy
import torch
import numpy as np
import os
import random
from torch.optim.lr_scheduler import _LRScheduler

def train(model, dataloader, optimizer, criterion, device):
    '''Trains the model for one epoch.
    Args:
        model (Model): Model.
        dataloader (DataLoader): DataLoader.
        optimizer (Optimizer): Optimizer.
        criterion (Criterion): Criterion.
        device (str): Device.
    Returns:
        float: Average training loss.
    '''
    model.train()
    loss_sum = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / len(dataloader)


def eval(model, dataloader, criterion, device):
    '''Evaluates the model.
    Args:
        model (Model): Model.
        dataloader (DataLoader): DataLoader.
        criterion (Criterion): Criterion.
        device (str): Device.
    Returns:
        float: Average validation loss.
        float: Accuracy.
    '''
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return val_loss / len(dataloader), correct / total



def set_seed(seed=2):
    '''Sets the seed for reproducibility.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed) 
    

class StepLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(StepLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 80:
            return [1e-3 for base_lr in self.base_lrs]
        elif 80 <= self.last_epoch < 120:
            return [1e-4 for base_lr in self.base_lrs]
        elif 120 <= self.last_epoch < 160:
            return [1e-5 for base_lr in self.base_lrs]
        elif 160 <= self.last_epoch < 180:
            return [1e-6 for base_lr in self.base_lrs]
        else:
            return [0.5e-6 for base_lr in self.base_lrs]

# sets the optimizer and the scheduler
def setup_optim(model, args, meta_net = False):
    setup = {}
    if args.data_set in ['cifar10', 'cifar100', 'svhn', 'celeb_a']:
        setup['optimizer'] = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
    elif args.data_set in ['tiny-imagenet']:
        setup['optimizer'] = torch.optim.SGD(model.parameters() , lr = 0.1, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
    if meta_net:
        setup['meta_optimizer'] = torch.optim.Adam(meta_net.parameters(), lr = args.meta_lr, weight_decay = args.meta_weight_decay)
    setup['lr'] = StepLRScheduler(setup['optimizer'], last_epoch=-1)
    return setup