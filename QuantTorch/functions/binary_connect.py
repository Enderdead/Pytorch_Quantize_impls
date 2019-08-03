import torch
from .common import front
from ..device import device
import warnings
warnings.simplefilter("always",DeprecationWarning)

"""
Implementation from Binary connect and Binary net :
https://arxiv.org/pdf/1602.02830.pdf
https://arxiv.org/pdf/1511.00363.pdf
"""


class BinaryConnectDeterministic(torch.autograd.Function):
    """
    Binarizarion deterministic op with backprob.\n
    Forward : \n
    :math:`r_b  = sign(r)`\n
    Backward : \n
    :math:`d r_b/d r = 1_{|r|=<1}`
    """
    @staticmethod
    def forward(ctx, input):
        """
        Apply stochastic binarization on input tensor.
        """
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the back propagation of the binarization op.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_output



class BinaryConnectStochastic(torch.autograd.Function):
    """
    Binarizarion stochastic op with backprob.\n
    Forward : \n
    :math:`r_b  = 1` with prob of :math:`hardsigmoid(r)`\n
    Backward : \n
    :math:`d r_b/d r = 1_{|r|=<1}`\n
    """
    @staticmethod
    def forward(ctx, input):
        """
        Apply stochastic binarization on input tensor.
        """
        ctx.save_for_backward(input)
        # z ~ uniform([0,1])
        z = torch.rand_like(input, requires_grad=False)
        # p = hard sigmoid(input)
        p = ((torch.clamp(input, -1, 1) + 1) / 2)
        # z<p = 1 with a probability of p  
        return -1.0 + 2.0 * (z<p).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the back propagation of the binarization op.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_input


def BinaryConnect(stochastic=False):
    """
    A torch.nn.Module is return with a Binarization op inside.
    Usefull on Sequencial instanciation.

    :param stochastic: Use Determinist or stochastic binarization.
    
    """
    act = BinaryConnectStochastic if stochastic else BinaryConnectDeterministic
    return front(act)


class BinaryDense(torch.autograd.Function):
    """
    Applies a linear transformation to the incoming data: y=W_b.x+b
    With W_b a binarized transformation of W with a deterministic way.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        weight_b = torch.sign(weight)
        # Apply classic linear op with quantified weight.
        output = torch.nn.functional.linear(input, weight_b, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        weight_b = torch.sign(weight)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output*Wb
            grad_input = grad_output.mm(weight_b)
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.T*input
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias



def BinaryConv2d(stride=1, padding=1, dilation=1, groups=1):
    """
    .. warning:: **DEPRECATED**
    
    Return a Conv2d Op with parameters given.
    Apply a Deterministic binarization on weight only.
    """
    warnings.warn("Deprecated conv op ! Huge cuda memory consumption due to torch.grad.cuda_grad.conv2d_input function.", DeprecationWarning,stacklevel=2)
    class _BinaryConv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            ctx.save_for_backward(input, weight, bias)
            weight_b = torch.sign(weight)
            output = torch.nn.functional.conv2d(input, weight_b, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_variables

            weight_b = torch.sign(weight)

            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(input.size(), weight_b, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0).sum(1).squeeze(1).sum(-1).squeeze(-1)

            if bias is not None:
                return grad_input, grad_weight, grad_bias
            else:
                return grad_input, grad_weight  

    return _BinaryConv2d



def AP2(x):
    """
    Return a power 2 approximation of x.\n
    return :\n
    .. math:: sign(x) × 2round(log2(x))\n

    Operator introduced here :  https://arxiv.org/pdf/1602.02830.pdf\n

    :param x: Tensor

    """
    two = torch.ones_like(x)*2
    return torch.sign(x) * torch.pow(two,torch.round(torch.log2(torch.abs(x))))



class ShiftBatch(torch.autograd.Function):
    """
    Primivite operator for batch normalizarion using shift operator instead of divide op.\n
    """
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, eps):
        inputs_mu = input - running_mean
        sqrtvar = torch.sqrt(running_var + eps)
        norm_inputs = inputs_mu*AP2(1/sqrtvar)
        out = norm_inputs*AP2(weight) + bias
        ctx.save_for_backward(input, weight, sqrtvar, norm_inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, sqrtvar, norm_inputs = ctx.saved_tensors

        grad_input = grad_running_mean = grad_running_var = grad_weight = grad_bias =  grad_eps = None
        if ctx.needs_input_grad[0]:
            # input_grad = grad_output*weight/sqrt(running_var + eps)
            grad_input = grad_output*weight
            grad_input = grad_input/sqrtvar

        if ctx.needs_input_grad[1]:
            # None because constant
            grad_running_mean = None

        if ctx.needs_input_grad[2]:
            # None because constant
            grad_running_var = None

        if ctx.needs_input_grad[3]:
            grad_weight = grad_output*norm_inputs

        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0).squeeze(0)

        if ctx.needs_input_grad[5]:
            #None because constant
            grad_eps = None
        
        return grad_input, grad_running_mean, grad_running_var, grad_weight, grad_bias,  grad_eps





