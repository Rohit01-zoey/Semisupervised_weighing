from models.resnetV2 import ResNetV2
from models.metanet import 

def get_model(str, *args, **kwargs):
    '''Returns the model.
    Args:
        str (str): Model name.
    Returns:
        Model: Model.
    '''
    model_arch = str.split('_')
    if model_arch[0] == 'rs' and model_arch[1] == 'v1':
        raise NotImplementedError
    if model_arch[0] == 'rs' and model_arch[1] == 'v2':
        return ResNetV2(int(model_arch[2]), *args, **kwargs)
    if model_arch[0] == 'rsmn' and model_arch[1] == 'v1':
        raise NotImplementedError
    else:
        raise ValueError('Model not found.')