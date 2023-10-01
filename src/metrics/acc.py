'''Implements the accuracy metric for the model'''
import torch

def accuracy(output, target):
    '''Computes the accuracy.
    Args:
        output (Tensor): Output of the model.
        target (Tensor): Ground truth.
    Returns:
        float: Accuracy.
    '''
    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    return correct / total

def maskedAccuracy(output, target):
    '''Computes the accuracy, ignoring samples with a target label of -1.
    Args:
        output (Tensor): Output of the model.
        target (Tensor): Ground truth.
    Returns:
        float: Accuracy.
    '''
    mask = (target != -1)
    _, predicted = torch.max(output.data[mask], 1)
    total = mask.sum().item()
    correct = (predicted == target[mask]).sum().item()
    return correct / total

def topkAccuracy(output, target, k):
    '''Computes the top-k accuracy.
    Args:
        output (Tensor): Output of the model.
        target (Tensor): Ground truth.
        k (int): Top-k.
    Returns:
        float: Top-k accuracy.
    '''
    _, predicted = torch.topk(output.data, k=k, dim=1)
    predicted = predicted.t()
    correct = predicted.eq(target.view(1, -1).expand_as(predicted))
    correct_k = correct[:k].view(-1).float().sum(0)
    total = target.size(0)
    return correct_k / total