import torch
from functions.common import front
from device import device

"""
Implementation from ternary connect :
https://arxiv.org/pdf/1510.03009.pdf
"""


class TernaryConnectDeterministic(torch.autograd.Function):
    """
        Ternary deterministic op.
        Forward : 
                  {1  if  x > 0.5
            x_t = {0  if |x|< 0.5
                  {-1 if  x <-0.5
        Backward : 
            d x_t / d x  = 1_{|r|=<1}
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        sign = torch.sign(input)
        return (sign + torch.sign(input -0.5*sign ))/2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_input


class TernaryConnectStochastic(torch.autograd.Function):
    """
        Ternary Stochastic op.
        Forward : 
            if x<0 :
                x_t = { 0 with prob of 1 + x
                      {-1 with prob of -x
            if x>=0:
                x_t = { 0 with prob of 1 - x 
                      { 1 with prob of x
        Backward : 
            d x_t / d x  = 1_{|r|=<1}
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        sign = torch.sign(input)
        z = torch.rand_like(input, requires_grad=False)
        #return sign - (z>torch.abs(input)).type(torch.FloatTensor).to(device) 
        return sign - torch.sign(z-torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_input



def TernaryConnect(stochastic=False):
    """
    A torch.nn.Module is return with a Ternary op inside.
    Usefull on Sequencial instanciation.
    """
    act = TernaryConnectStochastic if stochastic else TernaryConnectDeterministic
    return front(act)


def TernaryDense(stochastic=False):
    """
        Return a Linear op with Ternary quantization.
    """
    class _TernaryDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            sign = torch.sign(weight)
            if stochastic:
                z = torch.rand_like(weight, requires_grad=False)
                weight_t = sign - torch.sign(z-torch.abs(weight))
            else:
                weight_t = (sign + torch.sign(weight -0.5*sign ))/2
            
            ctx.save_for_backward(input, weight, weight_t, bias)
            output = torch.nn.functional.linear(input, weight_t, bias)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, weight_t, bias = ctx.saved_variables
            grad_input = grad_weight = grad_bias = None
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight_t)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
            return grad_input, grad_weight, grad_bias

    return _TernaryDense
    

        

def TernaryConv2d(stochastic=True, stride=1, padding=1, dilation=1, groups=1):
    """
        Return a Conv op with params given. Use Ternary to quantize weight before apply it.
    """
    class _TernaryConv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            sign = torch.sign(weight)
            if stochastic:
                z = torch.rand_like(weight, requires_grad=False)
                weight_t = sign - torch.sign(z-torch.abs(weight))
            else:
                weight_t = (sign + torch.sign(weight -0.5*sign ))/2

            ctx.save_for_backward(input, weight, weight_t, bias)
            output = torch.nn.functional.conv2d(input, weight_t, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, weight_t, bias = ctx.saved_variables

            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(input.size(), weight_t, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0).sum(1).squeeze(1).sum(-1).squeeze(-1)

            if bias is not None:
                return grad_input, grad_weight, grad_bias
            else:
                return grad_input, grad_weight  

    return _TernaryConv2d
