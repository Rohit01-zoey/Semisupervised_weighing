from models.resnetV2 import ResNetV2
from data.utils import fetch_dataset, split, prepare_online, get_dataloader
from metrics.loss import CrossEntropyLoss
from torchvision import transforms

import torch

def get_model(model_name, num_classes, seed = 2):
    '''Returns the model.
    Args:
        model_name (str): Name of the model. Currently supported: resnet_{9n+2}.
        num_classes (int): Number of classes.
    Returns:
        Model: Model.
    '''
    depth = int(model_name.split('_')[-1])
    return ResNetV2(depth=depth, num_classes=num_classes, seed = seed)


model = get_model('resnet_56', 100).to('cuda:4')

cifar = fetch_dataset('cifar100')

cifar['labeled'], cifar['unlabeled']= split(cifar['train'], split_ratio = 0.15, shuffle = True, seed = 2)

celoss = CrossEntropyLoss().to('cuda:4')


for epoch in range(100):
    tnx = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)) 
        #! , fill=(0, 0, 0), interpolation=InterpolationMode.BILINEAR)
    ])
    cifar['trnx:labeled']= prepare_online(cifar['labeled'], transform = tnx)
    
    dl =get_dataloader(cifar['trnx:labeled'], batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    for x,y in dl:
        loss=0
        x = x.to('cuda:4')
        y = y.to('cuda:4')
        out = model(x)
        loss = (1.0/128)*torch.sum(celoss(out, y))
        loss.backward()
    print(loss)