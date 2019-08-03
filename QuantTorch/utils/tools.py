import torch
import torch.nn as nn
from ..layers.common import QLayer
def to_oneHot(tensor, nb_classes=10):

    one_hot = torch.FloatTensor(tensor.size(0), nb_classes)
    try:
        one_hot = one_hot.to(tensor.get_device())
    except RuntimeError:
        pass
    one_hot.zero_()
    one_hot.scatter_(1, tensor.view(tensor.size(0),1).long() , 1)
    return one_hot

def iterable(obj):
    return '__iter__' in obj.__dir__()


def flat_net(model, class_to_get):
    result = list()
    if isinstance(model, class_to_get):
        return [model]
    for component in model.__dict__["_modules"].keys():
        if isinstance(model.__getattr__(component), class_to_get):
            result.append(model.__getattr__(component))
            continue
        if isinstance(model.__getattr__(component), torch.nn.Module):
            result += flat_net(model.__getattr__(component), class_to_get)
    return result


def weight_model(model, class_to_get, bitwidth_conf):
    result = 0
    for index, layer in enumerate(flat_net(model, class_to_get)):
        weight = layer.weight.data.cpu().detach().numpy().reshape(-1)
        result += weight*bitwidth_conf[index]
    return result


def replace_module(model, old_module, new_module):
    if model is old_module:
        return new_module
    
    for component in model.__dict__["_modules"].keys(): 
        if model.__getattr__(component) is old_module:
            model.__setattr__(component, new_module)
            break
        if isinstance(model.__getattr__(component), torch.nn.Module):
            model.__setattr__(component, replace_module(model.__getattr__(component),old_module, new_module))    
    return model

