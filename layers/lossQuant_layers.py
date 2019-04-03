import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt
from device import device
from layers.common import QLayer
from functions import loss_quant_connect

class LinearQuantLin(torch.nn.Module, QLayer):
    @staticmethod
    def convert(other, bottom=-1, top=1, size=5, alpha=1):
        if not isinstance(other, torch.nn.Linear):
            raise TypeError("Expected a torch.nn.Linear ! Receive:  {}".format(other.__class__))
        return LinearQuantLin(other.in_features, other.out_features, False if other.bias is None else True, bottom=bottom, top=top, size=size, alpha=alpha)

    def __init__(self, in_features, out_features, bias=True, bottom=-1, top=1, size=5, alpha=1):
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.bottom = bottom
        self.top = top
        self.register_buffer("alpha", torch.Tensor([alpha]).to(device))
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.linear_op = loss_quant_connect.QuantLinDense(size=size, bottom=bottom, top=top)

    def reset_parameters(self):
        self.weight.data.uniform_(self.bottom, self.top)
        if self.bias is not None:
            self.bias.data.zero_()

    def clamp(self):
        self.weight.data.clamp_(self.bottom, self.top)
        if not self.bias is None:
            self.bias.data.clamp_(self.bottom, self.top) 

    def set_alpha(self, alpha):
        self.alpha = torch.Tensor([alpha]).to(device)

    def train(self, mode=True):
        if self.training==mode:
            return
        if mode:
            self.weight.data.copy_(self.weight.org.data)
        else: # Eval mod
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.org.data.copy_(self.weight.data)
            self.weight.data.copy_(loss_quant_connect.lin_proj(self.weight, bottom=self.bottom, top=self.top, size=self.size).detach())

    def forward(self, input):
        return self.linear_op.apply(input, self.weight, self.bias, self.alpha)


class LinearQuantLog(torch.nn.Module, QLayer):
    @staticmethod
    def convert(other, gamma=2, init=0.25, size=5, alpha=1):
        if not isinstance(other, torch.nn.Linear):
            raise TypeError("Expected a torch.nn.Linear ! Receive:  {}".format(other.__class__))
        return LinearQuantLin(other.in_features, other.out_features, False if other.bias is None else True, gamma=gamma, init=init, size=size, alpha=alpha)

    def __init__(self, in_features, out_features, bias=True, gamma=2, init=0.25, size=5, alpha=1):
        super(LinearQuantLog, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma
        self.init = init
        self.size = size
        self.register_buffer("alpha", torch.Tensor([alpha]).to(device))
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.linear_op = loss_quant_connect.QuantLogDense(gamma=gamma, init=init, size=size)

    def reset_parameters(self):
        self.weight.data.uniform_(-self.init*self.gamma**(self.size-1), self.init*self.gamma**(self.size-1))
        if self.bias is not None:
            self.bias.data.zero_()

    def train(self, mode=True):
        if self.training==mode:
            return
        if mode:
            self.weight.data.copy_(self.weight.org.data)
        else: # Eval mod
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.org.data.copy_(self.weight.data)
            self.weight.data.copy_(loss_quant_connect.exp_proj(self.weight, gamma=self.gamma, init=self.init, size=self.size).detach())

    def clamp(self):
        self.weight.data.clamp_(-self.init*self.gamma**(self.size-1), self.init*self.gamma**(self.size-1))
        if not self.bias is None:
            self.bias.data.clamp_(-self.init*self.gamma**(self.size-1), self.init*self.gamma**(self.size-1))

    def set_alpha(self, alpha):
        self.alpha = torch.Tensor([alpha]).to(device)


    def forward(self, input):
        return self.linear_op.apply(input, self.weight, self.bias, self.alpha)


class QuantConv2dLin(torch.nn.Conv2d, QLayer):
    def __init__(self, *args,  bottom=-1, top=1, size=5, alpha=1,  stride=1, padding=1, dilation=1, groups=1):
        torch.nn.Conv2d.__init__(self, *args,  stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.top = top
        self.bottom = bottom
        self.size = size
        self.register_buffer("alpha", torch.Tensor([alpha]).to(device))
        self.conv_op = loss_quant_connect.QuantConv2d(size=self.size, bottom=self.bottom, top=self.top, stride=stride, padding=padding, dilation=dilation, groups=groups)


    def train(self, mode=True):
        if self.training==mode:
            return
        if mode:
            self.weight.data.copy_(self.weight.org.data)
        else: # Eval mod
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.org.data.copy_(self.weight.data)
            self.weight.data.copy_(loss_quant_connect.lin_proj(self.weight, bottom=self.bottom, top=self.top, size=self.size).detach())


    def reset_parameters(self):
        self.weight.data.uniform_(self.bottom, self.top)
        if self.bias is not None:
            self.bias.data.zero_()

    def clamp(self):
        self.weight.data.clamp_(self.bottom, self.top)
        if not self.bias is None:
            self.bias.data.clamp_(self.bottom, self.top) 

    def set_alpha(self, alpha):
        self.alpha = torch.Tensor([alpha]).to(device)

    def forward(self, input):
        out = self.conv_op.apply(input, self.weight, self.bias, self.alpha)
        return out