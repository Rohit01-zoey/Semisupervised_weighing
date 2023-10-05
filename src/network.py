from metrics.acc import accuracy
import torch
import numpy as np
import os
import random

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
    train_loss = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)


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