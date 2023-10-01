from metrics.acc import accuracy
import torch

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