import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt
from layers.common import QLayer
from functions import binary_connect

class LinearBin(torch.nn.Linear, QLayer):
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
            return torch.nn.functional.linear(input, binary_connect.BinaryConnectDeterministic.apply(self.weight), (self.bias))
        return torch.nn.functional.linear(input, binary_connect.BinaryConnectStochastic.apply(self.weight), (self.bias))


class BinConv2d(torch.nn.Conv2d, QLayer):

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

class ShiftNormBatch1d(torch.nn.Module):
    __constants__ = ['momentum', 'eps']
    def __init__(self, in_dim, eps=1e-5, momentum=0.1):
        super(ShiftNormBatch1d, self).__init__()
        self.in_features = in_dim
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(self.in_features))

        self.register_buffer('running_mean', torch.zeros(self.in_features))
        self.register_buffer('running_var', torch.ones(self.in_features))

        self.eps = eps
        self.momentum = momentum


    def forward(self, x):
        self.running_mean =  (1 - self.momentum) * self.running_mean + self.momentum * torch.mean(x,0).detach()
        self.running_var  =  (1 - self.momentum) * self.running_var  + self.momentum * torch.mean((x-self.running_mean)*binary_connect.AP2(x-self.running_mean),0).detach()

        return binary_connect.ShiftBatch.apply(x, self.running_mean, self.running_var, self.weight, self.bias, self.eps)




class ShiftNormBatch2d(torch.nn.Module):
    __constants__ = ['momentum', 'eps']
    def __init__(self, in_channels, eps=1e-5, momentum=0.1):
        super(ShiftNormBatch2d, self).__init__()
        self.in_features = in_channels
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(self.in_features))

        self.register_buffer('running_mean', torch.zeros(self.in_features))
        self.register_buffer('running_var', torch.ones(self.in_features))

        self.eps = eps
        self.momentum = momentum

    @staticmethod
    def _tile(tensor, dim):
        return tensor.repeat(dim[0],dim[1],1).transpose(2,0)

    def forward(self, x):
        dim = x.size()[-2:]

        self.running_mean =  (1 - self.momentum) * self.running_mean + self.momentum * torch.mean(x,[0,2,3]).detach()
        curr_mean = ShiftNormBatch2d._tile(self.running_mean, dim)

        self.running_var  =  (1 - self.momentum) * self.running_var  + self.momentum * torch.mean((x-curr_mean)*binary_connect.AP2(x-curr_mean),[0,2,3]).detach()

        return binary_connect.ShiftBatch.apply(x, curr_mean, ShiftNormBatch2d._tile(self.running_var, dim), ShiftNormBatch2d._tile(self.weight, dim), ShiftNormBatch2d._tile(self.bias, dim), self.eps)