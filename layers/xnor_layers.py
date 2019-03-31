import torch 
from layers.common import QLayer
from functions import xnor_connect
import warnings
warnings.simplefilter("always",DeprecationWarning)
warnings.warn("Module not finished due to  gradient implementation on conv layer !", ImportWarning)

class LinearXNOR(torch.nn.Linear, QLayer):
    def __init__(self, in_features, out_features, bias=True, dim=[0,1], **kwargs):
        super(LinearXNOR, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.lin_op = xnor_connect.XNORDense(dim=dim)


    def forward(self, input):
        return self.lin_op.apply(input, self.weight, self.bias)

class XNORConv2d(torch.nn.Conv2d, QLayer):

    def __init__(self, *kargs, dim=[0,1], quant_input=False, **kwargs):
        torch.nn.Conv2d.__init__(self, *kargs, **kwargs)
        self.conv_op =  xnor_connect.XNORConv2d(dim, False, self.stride, self.padding, self.dilation, self.groups)

    def clamp(self):
        pass

    def forward(self, input):
        out = self.conv_op.apply(input, self.weight, self.bias)
        return out


