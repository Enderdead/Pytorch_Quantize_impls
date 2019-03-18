import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt

from functions import terner_connect

class LinearTer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearTer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def clamp(self):
        self.weight.data.clamp_(-1, 1)
        if not self.bias is None:
            self.bias.data.clamp_(-1, 1) 

    def forward(self, input):
        return terner_connect.TernerDense(input, self.weight, self.bias, True)


class TernerConv2d(torch.nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(TernerConv2d, self).__init__(*kargs, **kwargs)

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.5)
        self.weight.data.clamp_(-1, 1)
        if self.bias is not None:
            self.bias.data.zero_()

    def clamp(self):
        self.weight.data.clamp_(-1, 1)
        #if not self.bias is None:
        #    self.bias.data.clamp_(-1, 1)


    def forward(self, input):
        out = terner_connect.TernerConv2d(input, self.weight, self.bias, True, self.stride,
                                   self.padding, self.dilation, self.groups)

        return out



