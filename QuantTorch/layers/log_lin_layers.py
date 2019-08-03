import torch 
from .common import QLayer
from ..functions import log_lin_connect


class LinearQuant(torch.nn.Linear, QLayer):
    @staticmethod
    def convert(other, dtype="lin", fsr=7, bitwight=3):
        if not isinstance(other, torch.nn.Linear):
            raise TypeError("Expected a torch.nn.Linear ! Receive:  {}".format(other.__class__))
        return LinearQuant(other.in_features, other.out_features, False if other.bias is None else True, dtype=dtype, fsr=fsr, bitwight=bitwight)

    def __init__(self, in_features, out_features, bias=True, dtype="lin", fsr=7, bitwight=3):
        self.bitwight = bitwight
        self.fsr = fsr
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.weight_op = log_lin_connect.nnQuant(dtype=dtype, fsr=fsr, bitwight=bitwight, with_sign=True, lin_back=True)

    def clamp(self):
        self.weight.data.clamp_(-1*2**(self.fsr), 2**(self.fsr))

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

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, 2**(self.fsr-self.bitwight), 2**(self.fsr))
        self.weight.data.mul_( (torch.rand_like(self.weight)<0.5).type(self.weight.dtype)*2-1)
        if not self.bias is None:
            self.bias.data.zero_()

    def forward(self, input):
        out = torch.nn.functional.linear(input, self.weight_op.forward(self.weight), self.bias)
        return out


class QuantConv2d(torch.nn.Conv2d, QLayer):

    @staticmethod
    def convert(other, fsr=7, bitwight=3, dtype="lin"):
        if not isinstance(other, torch.nn.Conv2d):
            raise TypeError("Expected a torch.nn.Conv2d ! Receive:  {}".format(other.__class__))
        return QuantConv2d(other.in_channels, other.out_channels, other.kernel_size, stride=other.stride,
                         padding=other.padding, dilation=other.dilation, groups=other.groups,
                         bias=False if other.bias is None else True, fsr=fsr, bitwight=bitwight, dtype=dtype)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, fsr=7, bitwight=3, dtype="lin"):

        self.fsr = fsr
        self.bitwight = bitwight
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.weight_op = log_lin_connect.nnQuant(dtype=dtype, fsr=fsr, bitwight=bitwight, with_sign=True, lin_back=True)

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

    def reset_parameters(self):
        if self.bitwight==32:
            super(QuantConv2d, self).reset_parameters()
        torch.nn.init.uniform_(self.weight, 2**(self.fsr-self.bitwight), 2**(self.fsr))
        self.weight.data.mul_( (torch.rand_like(self.weight)<0.5).type(self.weight.dtype)*2-1)
        if not self.bias is None:
            self.bias.data.zero_()

    def clamp(self):
        self.weight.data.clamp_(-1*2**(self.fsr), 2**(self.fsr))

    def forward(self, input):
        out = torch.nn.functional.conv2d(input, self.weight_op.forward(self.weight), bias=self.bias, stride=self.stride,
                                         padding=self.padding, dilation=self.dilation, groups=self.groups)
        return out



