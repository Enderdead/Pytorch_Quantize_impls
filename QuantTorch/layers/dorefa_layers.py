import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt
from .common import QLayer
from ..functions import dorefa_connect


class LinearDorefa(torch.nn.Linear, QLayer):
    @staticmethod
    def convert(other, weight_bit=3):
        if not isinstance(other, torch.nn.Linear):
            raise TypeError("Expected a torch.nn.Linear ! Receive:  {}".format(other.__class__))
        return LinearDorefa(other.in_features, other.out_features, False if other.bias is None else True, weight_bit=weight_bit)

    def __init__(self, in_features, out_features, bias=True, weight_bit=3):
        """
        Test
        """
        torch.nn.Linear.__init__(self, in_features, out_features,  bias=bias)
        self.bitwight = weight_bit
        self.weight_op = dorefa_connect.nnQuantWeight(bitwight=weight_bit)

    def extra_repr(self):
        return "bitwidth = {}".format(self.bitwight)

    def train(self, mode=True):
        if self.training==mode:
            return
        if mode:
            self.weight.data.copy_(self.weight.org.data)
        else: # Eval mod
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.org.data.copy_(self.weight.data)
            self.weight.data.copy_(self.weight_op.forward(self.weight).detach())

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight_op.forward(self.weight), self.bias)

class DorefaConv2d(torch.nn.Conv2d, QLayer):

    @staticmethod
    def convert(other, weight_bit=3):
        if not isinstance(other, torch.nn.Conv2d):
            raise TypeError("Expected a torch.nn.Conv2d ! Receive:  {}".format(other.__class__))
        return DorefaConv2d(other.in_channels, other.out_channels, other.kernel_size, stride=other.stride,
                         padding=other.padding, dilation=other.dilation, groups=other.groups,
                         bias=False if other.bias is None else True, weight_bit=weight_bit)


    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_bit=3):
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=True)
        self.weight_op = dorefa_connect.nnQuantWeight(bitwight=weight_bit)

    def train(self, mode=True):
        if self.training==mode:
            return
        if mode:
            self.weight.data.copy_(self.weight.org.data)
        else: # Eval mod
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.org.data.copy_(self.weight.data)
            self.weight.data.copy_(self.weight_op.forward(self.weight).detach())

    def forward(self, input):
        if self.training:
            out = torch.nn.functional.conv2d(input, self.weight_op.forward(self.weight), self.bias,self.stride, self.padding, self.dilation, self.groups)
        else:
            out = torch.nn.functional.conv2d(input, self.weight, self.bias,self.stride, self.padding, self.dilation, self.groups)
        return out
