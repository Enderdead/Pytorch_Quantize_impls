import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt

from functions import binary_connect

"""
class LinearBin(torch.nn.Linear):
    def __init__(self, *args, stochastic=False, **kwargs):
        super(LinearBin, self).__init__(*args, **kwargs)
        self.stochastic = stochastic

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, -1, 1)
        if not self.bias is None:
            torch.nn.init.uniform_(self.bias, -1 , 1)

    def clamp(self):
        self.weight.data.clamp_(-1, 1)
        if not self.bias is None:
            self.bias.data.clamp_(-1, 1)

    def forward(self, input):
        return torch.nn.functional.linear(input, binary_connect.BinaryConnectDeterministic.apply(self.weight),self.bias)

"""

class LinearBin(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearBin, self).__init__()
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
        if self.bias is not None:
            return binary_connect.BinaryDense.apply(input, self.weight, self.bias)
        return binary_connect.BinaryDense.apply(input, self.weight)


class BinarizeConv2d(torch.nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def clamp(self):
        self.weight.data.clamp_(-1, 1)
        #if not self.bias is None:
        #    self.bias.data.clamp_(-1, 1)


    def forward(self, input):
        out = binary_connect.BinaryConv2d(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)

        return out




class Conv2dBin(torch.nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dBin, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *_pair(kernel_size)))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()


    def clamp(self):
        self.weight.data.clamp_(-1, 1)
        if not self.bias is None:
            self.bias.data.clamp_(-1, 1)


    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, -1, 1)
        if not self.bias is None:
            torch.nn.init.uniform_(self.bias, -1 , 1)


    @weak_script_method
    def forward(self, input):
        return F.conv2d(input, binary_connect.BinaryConnectDeterministic.apply(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
