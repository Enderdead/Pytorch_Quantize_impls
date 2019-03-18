import torch 
from torch._jit_internal import weak_module, weak_script_method, List
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt

from functions import log_lin_connect


class LinearLinQuant(torch.nn.Linear):
    def __init__(self, *args, fsr=7, bitwight=3, **kwargs):
        super(LinearLinQuant, self).__init__(*args, **kwargs)
        self.fsr = fsr
        self.bitwight = bitwight

    def forward(self, input):
        return log_lin_connect.QuantDense(input, self.weight, self.bias, self.fsr, self.bitwight)



class QuantLinConv2d(torch.nn.Conv2d):
    def __init__(self, *kargs, fsr=7, bitwight=3, **kwargs):
        super(QuantLinConv2d, self).__init__(*kargs, **kwargs)
        self.fsr = fsr
        self.bitwight = bitwight

    @weak_script_method
    def forward(self, input):
        out = log_lin_connect.QuantConv2d(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups, self.fsr, self.bitwight)

        return out



