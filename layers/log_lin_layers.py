import torch 
from layers.common import QLayer
from functions import log_lin_connect


class LinearQuant(torch.nn.Linear, QLayer):
    def __init__(self, in_features, out_features, bias=True, dtype="lin", fsr=7, bitwight=3):
        self.bitwight = bitwight
        self.fsr = fsr
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.weight_op = log_lin_connect.nnQuant(dtype=dtype, fsr=fsr, bitwight=bitwight, with_sign=True, lin_back=True)

    def clamp(self):
        self.weight.data.clamp_(-1*2**(self.fsr), 2**(self.fsr))

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, 2**(self.fsr-self.bitwight), 2**(self.fsr))
        self.weight.data.mul_( (torch.rand_like(self.weight)<0.5).type(self.weight.dtype)*2-1)
        if not self.bias is None:
            self.bias.data.zero_()
    def forward(self, input):
        out = torch.nn.functional.linear(input, self.weight_op.forward(self.weight), self.bias)
        return out


class QuantConv2d(torch.nn.Conv2d, QLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, fsr=7, bitwight=3, dtype="lin"):

        self.fsr = fsr
        self.bitwight = bitwight
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.weight_op = log_lin_connect.nnQuant(dtype=dtype, fsr=fsr, bitwight=bitwight, with_sign=True, lin_back=True)

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



