import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, target, advice = None):
        if advice is not None:
            return torch.mul(advice, self.loss_fn(output, target))
        else:
            return self.loss_fn(output, target)

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, output, target):
        mask = (target != -1)
        output = output[mask]
        target = target[mask]
        return self.loss_fn(output, target)

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.loss_fn = nn.KLDivLoss(reduction='None')

    def forward(self, output, target, advice = None):
        if advice is not None:
            return torch.mul(advice, self.loss_fn(torch.log(output), target))
        else:
            return self.loss_fn(torch.log(output), target)