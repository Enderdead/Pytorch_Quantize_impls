import torch 
from .common import QLayer
from ..functions import xnor_connect
import warnings
warnings.simplefilter("always",DeprecationWarning)
warnings.warn("Module not finished due to  gradient implementation on conv layer !", ImportWarning)

class LinearXNOR(torch.nn.Linear, QLayer):
    @staticmethod
    def convert(other, dim=[0,1]):
        if not isinstance(other, torch.nn.Linear):
            raise TypeError("Expected a torch.nn.Linear ! Receive:  {}".format(other.__class__))
        return LinearXNOR(other.in_features, other.out_features, False if other.bias is None else True, dim=dim)

    def __init__(self, in_features, out_features, bias=True, dim=[0,1]):
        super(LinearXNOR, self).__init__(in_features, out_features, bias=bias)
        self.lin_op = xnor_connect.XNORDense(dim=dim)
        self.dim = dim

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
            self.weight.data.copy_(torch.mean(torch.abs(self.weight), dim, keepdim=True)*torch.sign(self.weight).detach())


    def forward(self, input):
        return self.lin_op.apply(input, self.weight, self.bias)

class XNORConv2d(torch.nn.Conv2d, QLayer):

    @staticmethod
    def convert(other, dim=[0,1], quant_input=False):
        if not isinstance(other, torch.nn.Conv2d):
            raise TypeError("Expected a torch.nn.Conv2d ! Receive:  {}".format(other.__class__))
        return XNORConv2d(other.in_channels, other.out_channels, other.kernel_size, stride=other.stride,
                         padding=other.padding, dilation=other.dilation, groups=other.groups,
                         bias=False if other.bias is None else True, dim=dim, quant_input=quant_input)


    def __init__(self, *kargs, dim=[0,1], quant_input=False, **kwargs):
        torch.nn.Conv2d.__init__(self, *kargs, **kwargs)
        self.conv_op =  xnor_connect.XNORConv2d(dim, False, self.stride, self.padding, self.dilation, self.groups)

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
            self.weight.data.copy_(torch.mean(torch.abs(self.weight), dim, keepdim=True)*torch.sign(self.weight).detach())


    def clamp(self):
        pass

    def forward(self, input):
        out = self.conv_op.apply(input, self.weight, self.bias)
        return out


