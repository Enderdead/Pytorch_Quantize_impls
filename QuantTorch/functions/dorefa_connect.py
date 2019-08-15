import torch
from .common import front, safeSign
import warnings
warnings.simplefilter("always",DeprecationWarning)
"""
Implementation of Dorefa net :
https://arxiv.org/pdf/1606.06160.pdf
"""


def _quantize(x, bit_width=3):
    """
    Quantize operator used in all quantization of Dorefa net.

    :math:`quantize(x) =  round((2^k -1)x)/(2^k -1)`
    with x in [0,1]  and quantize(x) in [0,1]
    and with k = bits
    """
    if bit_width==1:
        return safeSign(x)
    elif bit_width==32:
        return x
    else:
        two = torch.ones_like(x)*2
        return ((1)/(torch.pow(two,bit_width)-1))*torch.round((torch.pow(two,bit_width)-1)*x)


def nnDorefaQuant(bit_width=3):
    """
    Return a Torch.nn.Module fronter with a Quant operator inside.\n
    :param bitwight: Number of bits to use for quantization op.
    :param with_sign: Add a sign bit or not.
    Forward : \n
    :math:`quantize(x)` see _quantize() \n
    Backward : \n
    :math:`dx_q/dx  = 1`\n
    """
    class _Quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return _quantize(input, bit_width=bit_width)

        def backward(ctx, grad_ouput):
            return grad_ouput.clone()
    return front(_Quant)
   


def DorefaQuant(x, bit_width=3):
    """
    Apply a quantize op on x.
    :param bit_width: Number of bits to use for quantization op.
    :param with_sign: Add a sign bit or not.
    """
    class _Quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return _quantize(input, bit_width=bit_width)

        @staticmethod
        def backward(ctx, grad_ouput):
            return grad_ouput.clone()
    return _Quant.apply(x)


class _ignore_factor_op(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, const):
            return input*const

        @staticmethod
        def backward(ctx, grad_ouput):
            var_grad = None
            const_grad = None
            if ctx.needs_input_grad[0]:
                var_grad = grad_ouput.clone()
            if ctx.needs_input_grad[1]:
                const_grad = 0
            return var_grad, const_grad


def nnQuantWeight(bit_width=3):
    r"""
    Return a Module fronter with quantize op for layer's weight.
    This Op include take any real weight range.

    :param bitwight: Number of bits to use for quantization op (without bit sign)

    Forward : 
    :math:`2*quantize_{k}(\frac{ tanh(W)}{2.max( | tanh(W) |)}+\frac{1}{2} )-1`   
    """

    class _QuantWeight(torch.nn.Module):
        def __init__(self):
            super(_QuantWeight, self).__init__()
            self.bit_width = bit_width
            self.quant_op =  nnDorefaQuant(bit_width)

        def forward(self, x):
            if self.bit_width==1:
                E = torch.mean(torch.abs(x)).detach()
                weight = _ignore_factor_op.apply(self.quant_op(x) , E )
            elif self.bit_width==32:
                return x
            else:
                if torch.max(torch.abs(x)) ==0.0:
                    return torch.zeros_like(x)
                weight = torch.tanh(x)
                weight = weight / (2 * torch.max(torch.abs(weight)) ) + 0.5
                weight = 2 * self.quant_op(weight) - 1
            return weight

    return _QuantWeight()


def QuantDense(bit_width=3):
    """ 
    .. warning:: **DEPRECATED**
    Applies a linear transformation to the incoming data: y=W_b.x+b
    With W_b a quantized transformation of W.

    :param bit_width: Number of bits to use for quantization op (without bit sign)
    """
    class _QuantDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            max_abs = torch.max(torch.abs(torch.tanh(weight)))
            if bit_width==1:
                weight_q = safeSign(weight)* torch.mean(torch.abs(weight)).detach()
            elif bit_width==32:
                weight_q = weight
            else:
                weight_q = 2*_quantize(0.5  + torch.tanh(weight)/(2*max_abs)   , bit_width=bit_width) - 1
            
            output = torch.nn.functional.linear(input, weight_q, bias)
            ctx.save_for_backward(input, weight, weight_q, max_abs, bias)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, weight_q, max_abs, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight_q)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
                if bit_width==1 or bit_width==32:
                    grad_weight = grad_weight
                else:
                    grad_weight =  grad_weight * (1 - torch.pow(torch.tanh(weight),2)) / max_abs
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
            return grad_input, grad_weight, grad_bias

    return _QuantDense


def QuantConv2d(stride=1, padding=1, dilation=1, groups=1, bit_width=3):
    """
    .. warning:: **DEPRECATED**
    Return a Conv2d op with settings given.
    If bit_width=1, return a XNOR like op.
    """
    warnings.warn("Deprecated conv op ! Huge cuda memory consumption due to torch.grad.cuda_grad.conv2d_input function.", DeprecationWarning,stacklevel=2)
    class _QuantConv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            max_weight = torch.max(torch.abs(weight))
            if bit_width==1:
                weight_q = safeSign(weight)* torch.mean(torch.abs(weight)).detach()
            elif bit_width==32:
                weight_q = weight
            else:
                weight_q = 2*_quantize(0.5  + torch.tanh(weight)/(2*torch.tanh(max_weight))   , bit_width=bit_width) - 1
            
            ctx.save_for_backward(input, weight, weight_q, max_weight, bias)
            output = torch.nn.functional.conv2d(input, weight_q, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, weight_q, max_weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(input.size(), weight_q, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
                if bit_width>1 and bit_width<32:
                    grad_weight = grad_weight * (1 - torch.pow(torch.tanh(weight),2)) /  torch.tanh(max_weight) 
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0).sum(1).squeeze(1).sum(-1).squeeze(-1)

            if bias is not None:
                return grad_input, grad_weight, grad_bias
            else:
                return grad_input, grad_weight  

    return _QuantConv2d

