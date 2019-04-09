import torch
from functions.common import front
from device import device
"""
Implementation of Convolutional Neural Networks using Logarithmic Data Representation :
https://arxiv.org/pdf/1603.01025.pdf
"""

def LogQuant(fsr=7, bitwight=3, with_sign=True, lin_back=True):
    r"""
        Generate a Quantization op using Log method from Imp. CNN using Log Data Rep.

        :param fsr: Max value of the output.
        :param bitwight:  Numbers of bits on this quant op.
        :param with_sign: Add a sign bit to quant op.
        :param lin_back: Use linear back propagation or a quantized gradient.

        Forward:

        :math:`Quant(x) = 2^{clamp(round(ln_2(|x|)},fsr-2^{bitwight}, fsr))`

        BackWard (if not lin_back):

        :math:`grad\_input = sign(grad_output)* 2^{clamp(round(ln_2(|grad\_output|)},fsr-2^{bitwight}, fsr))`

    """
    class _LogQuant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if with_sign:
                return torch.sign(input)*torch.pow(torch.ones_like(input)*2, torch.clamp(torch.round(torch.log2(torch.abs(input))), fsr-2**bitwight ,fsr )) 
            return torch.pow(torch.ones_like(input)*2, torch.clamp(torch.round(torch.log2(torch.abs(input))), fsr-2**bitwight ,fsr )) 

        @staticmethod
        def backward(ctx, grad_output):
            if lin_back:
                grad_input = grad_output.clone()
                return grad_input
            return torch.sign(grad_output) * torch.pow(torch.ones_like(grad_output)*2, torch.clamp(torch.round(torch.log2(torch.abs(grad_output))), fsr-2**bitwight,fsr )) 
    return _LogQuant

def LinQuant(fsr=7, bitwight=3, with_sign=True, lin_back=True):
    """
    Generate a Quantization op using Lin method from Imp. CNN using Log Data Rep.

    :param fsr: Max value of the output.
    :param bitwight:  Numbers of bits on this quant op.
    :param with_sign: Add a sign bit to quant op.
    :param lin_back: Use linear back propagation or a quantized gradient.
    
    Forward :

    :math:`Quant(x) = Clamp(Round(x/step)*step,0,2^{FSR}) with step = 2^{FSR-bitwight}`
    
    BackWard (if not lin_back):

    :math:`grad\_input = sign(grad_output)* Clamp(Round(grad\_output/step)*step,0,2^{FSR})`
    """
    class _LinQuant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # Clip(Round(x/step)*step,0,2^(FSR)) with step = 2^{FSR-bitwight}
            if(bitwight==32):
                return input
            step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
            if with_sign:
                return torch.sign(input)*torch.clamp(torch.round(torch.abs(input)/step)*step, 0,2**fsr)  
            return torch.clamp(torch.round(input/step)*step, 0,2**fsr)  

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if bitwight==32:
                return grad_input
            step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
            if lin_back:
                return grad_input
            else:
                return torch.sign(grad_output)*torch.clamp(torch.round(grad_output/step)*step, 0,torch.FloatTensor([2]).pow(fsr).to(device) ) 
    return _LinQuant



def nnQuant(dtype="lin", fsr=7, bitwight=3, with_sign=True, lin_back=True):
    """
    Return a Torch Module fronter with Quantization op inside. Suport Lin and Log quantization.

    :param dtype: Use \'lin\' or \'log\' method.
    :param fsr: Max value of the output.
    :param bitwight:  Numbers of bits on this quant op.
    :param with_sign: Add a sign bit to quant op.
    :param lin_back: Use linear back propagation or a quantized gradient.

    """
    if dtype == "lin":
        return front(LinQuant(fsr=fsr, bitwight=bitwight, with_sign=with_sign, lin_back=lin_back))
    elif dtype=="log":
        return front(LogQuant(fsr=fsr, bitwight=bitwight, with_sign=with_sign, lin_back=lin_back))
    else:
        raise RuntimeError("Only \'log\' and \'lin\' dtype are supported !")


def Quant(input, dtype="lin", fsr=7, bitwight=3, with_sign=True, lin_back=True):
    """
    Apply a quantization with backprob support on input tensor.
    
    :param dtype: Use \'lin\' or \'log\' method.
    :param fsr: Max value of the output.
    :param bitwight:  Numbers of bits on this quant op.
    :param with_sign: Add a sign bit to quant op.
    :param lin_back: Use linear back propagation or a quantized gradient.
    """
    if dtype=="lin":
        return LinQuant(fsr=fsr,bitwight=bitwight,with_sign=with_sign, lin_back=lin_back).apply(input)
    elif dtype=="log":
        return LogQuant(fsr=fsr,bitwight=bitwight,with_sign=with_sign, lin_back=lin_back).apply(input)
    else:
        raise RuntimeError("Only \'log\' and \'lin\' dtype are supported !")

"""
# TODO Remove
def QuantDense(input, weight, bias=None, fsr=7, bitwight=3):
    class LinQuantDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
            ctx.save_for_backward(input, weight, bias)
            weight_q = torch.sign(weight)*torch.pow(torch.ones_like(weight)*2, torch.clamp(torch.round(torch.log2(torch.abs(weight))), fsr-2**bitwight ,fsr )) 
            output = torch.nn.functional.linear(input, weight_q, bias)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            weight_b = torch.sign(weight)
            grad_input = grad_weight = grad_bias = None
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight_b)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
            return grad_input, grad_weight, grad_bias

    return LinQuantDense.apply(input, weight, bias)


# TODO version log 
def QuantConv2d(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1, fsr=7, bitwight=3):
    class _QuantConv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
            ctx.save_for_backward(input, weight, bias)
            weight_q = torch.sign(weight)*torch.pow(torch.ones_like(weight)*2, torch.clamp(torch.round(torch.log2(torch.abs(weight))), fsr-2**bitwight ,fsr )) 
            output = torch.nn.functional.conv2d(input, weight_q, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
            weight_q = torch.sign(weight)*torch.pow(torch.ones_like(weight)*2, torch.clamp(torch.round(torch.log2(torch.abs(weight))), fsr-2**bitwight ,fsr )) 

            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(input.size(), weight_q, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0).sum(1).squeeze(1).sum(-1).squeeze(-1)

            if bias is not None:
                return grad_input, grad_weight, grad_bias
            else:
                return grad_input, grad_weight  

    return _QuantConv2d.apply(input, weight, bias)
"""