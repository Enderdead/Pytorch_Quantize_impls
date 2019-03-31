import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt
from layers.common import QLayer
from functions import terner_connect

class LinearTer(torch.nn.Linear, QLayer):
    def __init__(self, in_features, out_features, bias=True, deterministic=True):
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.deterministic = deterministic

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def clamp(self):
        self.weight.data.clamp_(-1, 1)
        if not self.bias is None:
            self.bias.data.clamp_(-1, 1) 

    def forward(self, input):
        if self.deterministic:
            return torch.nn.functional.linear(input, terner_connect.TernaryConnectDeterministic.apply(self.weight), (self.bias))
        return torch.nn.functional.linear(input, terner_connect.TernaryConnectStochastic.apply(self.weight), (self.bias))

class TerConv2d(torch.nn.Conv2d, QLayer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, deterministic=True):
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.deterministic = deterministic

    def clamp(self):
        self.weight.data.clamp_(-1, 1)

    def forward(self, input):
        if self.deterministic:
            return torch.nn.functional.conv2d(input, binary_connect.BinaryConnectDeterministic.apply(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return torch.nn.functional.conv2d(input, binary_connect.BinaryConnectStochastic.apply(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)


