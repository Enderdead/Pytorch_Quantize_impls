import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt
from .common import QLayer
from ..functions import terner_connect

class LinearTer(torch.nn.Linear, QLayer):
    @staticmethod
    def convert(other, dtype="lin", deterministic=True):
        if not isinstance(other, torch.nn.Linear):
            raise TypeError("Expected a torch.nn.Linear ! Receive:  {}".format(other.__class__))
        return LinearTer(other.in_features, other.out_features, False if other.bias is None else True, deterministic=deterministic)

    def __init__(self, in_features, out_features, bias=True, deterministic=True):
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.deterministic = deterministic
        if self.deterministic:
            self.ter_op = terner_connect.TernaryConnectDeterministic
        else:
            self.ter_op = terner_connect.TernaryConnectStochastic

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def train(self, mode=True):
        if self.training==mode:
            return
        self.training=mode
        if mode:
            self.weight.data.copy_(self.weight.org.data)
        else: # Eval mod
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.org.data.copy_(self.weight.data)
            self.weight.data.copy_(self.ter_op.apply(self.weight).detach())

    def clamp(self):
        self.weight.data.clamp_(-1, 1)
        if not self.bias is None:
            self.bias.data.clamp_(-1, 1) 

    def forward(self, input):
        if self.training:
            return torch.nn.functional.linear(input, self.ter_op.apply(self.weight), (self.bias))
        else:
            return torch.nn.functional.linear(input, self.weight, (self.bias))


class TerConv2d(torch.nn.Conv2d, QLayer):
    @staticmethod
    def convert(other, deterministic=True):
        if not isinstance(other, torch.nn.Conv2d):
            raise TypeError("Expected a torch.nn.Conv2d ! Receive:  {}".format(other.__class__))
        return TerConv2d(other.in_channels, other.out_channels, other.kernel_size, stride=other.stride,
                         padding=other.padding, dilation=other.dilation, groups=other.groups,
                         bias=False if other.bias is None else True, deterministic=deterministic)


    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, deterministic=True):
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.deterministic = deterministic
        if self.deterministic:
            self.ter_op = terner_connect.TernaryConnectDeterministic
        else:
            self.ter_op = terner_connect.TernaryConnectStochastic

    def clamp(self):
        self.weight.data.clamp_(-1, 1)

    def train(self, mode=True):
        if self.training==mode:
            return
        self.training=mode
        if mode:
            self.weight.data.copy_(self.weight.org.data)
        else: # Eval mod
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.org.data.copy_(self.weight.data)
            self.weight.data.copy_(self.ter_op.apply(self.weight).detach())

    def forward(self, input):
        if self.training:
            return torch.nn.functional.conv2d(input, self.ter_op.apply(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


