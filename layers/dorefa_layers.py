import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt

from functions import dorefa_connect


class LinearDorefa(torch.nn.Linear):
    def __init__(self, *args, bitwight=3, **kwargs):
        super(LinearDorefa, self).__init__(*args, **kwargs)
        self.bitwight = bitwight

    def forward(self, input):
        return dorefa_connect.QuantDense(input, self.weight, self.bias, self.bitwight)


class DorefaConv2d(torch.nn.Conv2d):
    def __init__(self, *kargs, bitwight=3, **kwargs):
        super(DorefaConv2d, self).__init__(*kargs, **kwargs)
        self.bitwight = bitwight

    @weak_script_method
    def forward(self, input):
        out = dorefa_connect.QuantConv2d(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups, self.bitwight)

        return out



