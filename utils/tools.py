import torch

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