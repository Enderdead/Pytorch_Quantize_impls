import torch 
import torch.nn as nn
from copy import deepcopy

# Import Binary net
from layers.binary_layers import LinearBin, ShiftNormBatch1d, BinConv2d

# Import Dorefa net
from layers.dorefa_layers import DorefaConv2d, LinearDorefa

# Import Xnor net
from layers.xnor_layers import XNORConv2d, LinearXNOR
 
# Import log lin net
from layers import log_lin_layers

# Import lass reg quant 
from layers import lossQuant_layers


def _convert_net(module, dict_replace):
    if len(list(module.modules()))==1 or module.__class__ in dict_replace.keys():
        fragment = list(module.modules())[0]
        if fragment.__class__ in dict_replace.keys():
            if isinstance(dict_replace.get(fragment.__class__), tuple):
                return dict_replace.get(fragment.__class__)[0].convert(fragment,**dict_replace.get(fragment.__class__)[1])
            else:
                return dict_replace.get(fragment.__class__).convert(fragment,**dict_replace.get(fragment.__class__))
        else:
            return module
    for component in module.__dict__["_modules"].keys():
        if isinstance(module.__getattr__(component), torch.nn.Module):
            module.__setattr__(component, _convert_net(module.__getattr__(component), dict_replace))
    return module


def convert(module, replace_dict):
    return _convert_net(deepcopy(module), replace_dict)

def binary_net_convert(net, deterministic=True):
    kwargs = {"deterministic":deterministic}
    dict_patch =  {nn.Linear : (LinearBin, kwargs),
                   nn.Conv2d : (BinConv2d, kwargs)}
    return _convert_net(deepcopy(net), dict_patch)


def dorefa_net_convert(net, weight_bit=3):
    kwargs = {"weight_bit": weight_bit}
    dict_patch =  {nn.Linear : (LinearDorefa, kwargs),
                   nn.Conv2d : (DorefaConv2d, kwargs)}
    return _convert_net(deepcopy(net), dict_patch)


def xnor_net_convert(net, dim=[0,1], quant_input=False):
    kwargs = {"dim": dim, "quant_input": quant_input}
    dict_patch = {nn.Linear : (LinearXNOR, kwargs),
                  nn.Conv2d : (XNORConv2d, kwargs)}
    return _convert_net(deepcopy(net), dict_patch) 

def log_lin_net_convert(net, fsr=7, bitwight=3, dtype="lin"):
    kwargs = { "fsr": fsr, "bitwight": bitwight, "dtype": dtype}
    dict_patch = {nn.Linear : (log_lin_layers.LinearQuant, kwargs),
                  nn.Conv2d : (log_lin_layers.QuantConv2d, kwargs)}
    return _convert_net(deepcopy(net), dict_patch) 

def loss_quant_log_convert(net, gamma=2, init=0.25, size=5, alpha=1):
    kwargs = {"gamma": gamma, "init": init, "size": size, "alpha": alpha}
    dict_patch = {nn.Linear : (lossQuant_layers.LinearQuantLog, kwargs),
                  nn.Conv2d : (lossQuant_layers.QuantConv2dLog, kwargs)}
    return _convert_net(deepcopy(net), dict_patch) 

def loss_quant_lin_convert(net, bottom=-1, top=1, size=5, alpha=1):
    kwargs = {"bottom": bottom, "top": top, "size": size, "alpha": alpha}
    dict_patch = {nn.Linear : (lossQuant_layers.LinearQuantLin, kwargs),
                  nn.Conv2d : (lossQuant_layers.QuantConv2dLin, kwargs)}
    return _convert_net(deepcopy(net), dict_patch) 
