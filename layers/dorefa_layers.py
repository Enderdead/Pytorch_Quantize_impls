import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt

from functions import dorefa_connect


class LinearDorefa(torch.nn.Linear):
    def __init__(self, *args, weight_bit=3, **kwargs):
        super(LinearDorefa, self).__init__(*args, **kwargs)
        self.bitwight = weight_bit
        self.lin_op = dorefa_connect.QuantDense(self.bitwight)

    def forward(self, input):
        return self.lin_op(input, self.weight, self.bias)


class DorefaConv2d(torch.nn.Conv2d):
    def __init__(self, *kargs, weight_bit=3, **kwargs):
        super(DorefaConv2d, self).__init__(*kargs, **kwargs)
        self.bitwight = weight_bit
        self.conv_op =  dorefa_connect.QuantConv2d(self.stride, self.padding, self.dilation, self.groups, self.bitwight)

        

    @weak_script_method
    def forward(self, input):
        out = self.conv_op(input, self.weight, self.bias)
        return out



