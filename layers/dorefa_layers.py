import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt
from layers.common import QLayer
from functions import dorefa_connect


class LinearDorefa(torch.nn.Linear, QLayer):
    def __init__(self, in_features, out_features, bias=True, weight_bit=3):
        torch.nn.Linear.__init__(self, in_features, out_features,  bias=bias)
        self.bitwight = weight_bit
        self.weight_op = dorefa_connect.nnQuantWeight(bitwight=weight_bit)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight_op.forward(self.weight), self.bias)

class DorefaConv2d(torch.nn.Conv2d, QLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_bit=3):
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=True)
        self.weight_op = dorefa_connect.nnQuantWeight(bitwight=weight_bit)

    def forward(self, input):
        out = torch.nn.functional.conv2d(input, self.weight_op.forward(self.weight), self.bias,self.stride, self.padding, self.dilation, self.groups)
        return out
