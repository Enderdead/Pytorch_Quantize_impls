import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt

from functions import xnor_connect

class LinearXNOR(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, dim=[0,1]):
        super(LinearXNOR, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.lin_op = xnor_connect.XNORDense(dim=dim)

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def clamp(self):
        pass
        #self.weight.data.clamp_(-1, 1)
        #if not self.bias is None:
        #    self.bias.data.clamp_(-1, 1) 

    def forward(self, input):
        return self.lin_op.apply(input, self.weight, self.bias)



class XNORConv2d(torch.nn.Conv2d):

    def __init__(self, *kargs, dim=[0,1], quant_input=False, **kwargs):
        super(XNORConv2d, self).__init__(*kargs, **kwargs)
        self.conv_op =  xnor_connect.XNORConv2d(dim, False, self.stride, self.padding, self.dilation, self.groups)

    def clamp(self):
        pass

    def forward(self, input):
        out = self.conv_op.apply(input, self.weight, self.bias)
        return out


