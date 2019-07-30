import torch
from functions.common import front
from functions.elastic_quant_connect import lin_deriv_l1, exp_deriv_l1
from device import device
import warnings
warnings.simplefilter("always",DeprecationWarning)
"""
Implementation from this paper :
https://publik.tuwien.ac.at/files/publik_275437.pdf
"""





def lin_deriv_WQR(x, kapa,top=1, bottom=-1, size=5):
    delta = (top-bottom)/(size-1)
    res = torch.zeros_like(x)
    for i in range(size):
        res -= kapa*(torch.sign(x)*torch.abs(x-(bottom+(i)*delta)) + torch.abs(x)*torch.sign(x-(bottom+(i)*delta)) ) *\
                 (x>(bottom+(i)*delta- delta/2)).float()*(x<(bottom+(i)*delta+ delta/2)).float()
    return res


def exp_deriv_WQR(x, kapa, gamma=2, init=0.25/2, size=5):
    res = torch.zeros_like(x)

    res -= kapa*(torch.sign(x)*torch.abs(x-(init)) + torch.abs(x)) *\
           (x>0).float()*(x<   ((init+init*gamma)/2) ).float()

    res -= kapa*(torch.sign(x)*torch.abs(x+init) + -1*torch.abs(x)) *\
           (x<0).float()*(x<   ((-init-init*gamma)/2) ).float()

    cur = init
    for _ in range(size-1):
        previous = cur
        cur *=gamma
        res -= kapa*(torch.sign(x)*torch.abs(x-cur) + torch.abs(x) ) *(x > (cur + previous) / 2).float()*(x < (cur+cur*gamma)/2).float()
        res -= kapa*(torch.sign(x)*torch.abs(x+cur) + torch.abs(x) ) *(x < (-cur - previous) / 2).float()*(x > (-cur-cur*gamma)/2).float()
    return res



def QuantWeightLin(top=1,  bottom=-1, size=5):
    class _QuantWeightOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, kapa, beta):
            ctx.save_for_backward(weight, kapa, beta)
            return weight
        
        @staticmethod
        def backward(ctx, output_grad):
            weight, kapa, beta = ctx.saved_tensors
            input_grad  = output_grad.clone()
            input_grad -= lin_deriv_WQR(weight, kapa,  top,  bottom, size)
            input_grad -= lin_deriv_l1(weight, beta,  top,  bottom, size)
            return input_grad, None, None
    return _QuantWeightOp

def QuantWeightExp(gamma=2, init=0.25, size=5):
    class _QuantWeightOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, kapa, beta):
            ctx.save_for_backward(weight, kapa, beta)
            return weight
        
        @staticmethod
        def backward(ctx, output_grad):
            weight, kapa, beta = ctx.saved_tensors
            input_grad  = output_grad.clone()
            input_grad -= exp_deriv_WQR(weight, kapa, gamma, init, size)
            input_grad -= exp_deriv_l1(weight, beta, gamma, init, size)

            return input_grad, None
    return _QuantWeightOp



def QuantLinDense(size=5, bottom=-1, top=1):
    """
    Return a linear transformation op with this form: y=W.x+b
    This operation inclue a backprob penality using Linear Quant method.
    """
    class _QuantLinDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias, kapa, beta):
            kapa.to(weight.device)
            beta.to(weight.device)
            ctx.save_for_backward(input, weight, bias, kapa, beta)
            return torch.nn.functional.linear(input, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, kapa, beta = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)

            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
                grad_weight -= lin_deriv_WQR(weight, kapa, bottom=bottom, top=top, size=size)
                grad_weight -= lin_deriv_l1(weight, beta=beta, bottom=bottom, top=top, size=size)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
                grad_bias -= lin_deriv_WQR(bias, kapa, bottom=bottom, top=top, size=size)
                grad_bias -= lin_deriv_l1(bias, beta=beta, bottom=bottom, top=top, size=size)

            return grad_input, grad_weight, grad_bias, None, None
    return _QuantLinDense


def QuantLogDense(gamma=2, init=0.25, size=5):
    """
        Return a quantization op with Log method.
        Quantized values are computed using geometrical sequence with init value = [init, -init] and scale factor = gamma.  

        param: 
            gamma
    """
    class _QuantLogDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias, kapa, beta):
            kapa.to(weight.device)
            beta.to(weight.device)
            ctx.save_for_backward(input, weight, bias, kapa, beta)
            return torch.nn.functional.linear(input, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, kapa, beta = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)

            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
                grad_weight -= exp_deriv_WQR(weight, kapa, gamma=2, init=0.25, size=size)
                grad_weight -= exp_deriv_l1(weight, beta=beta, gamma=2, init=0.25, size=size)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
                grad_bias -= exp_deriv_WQR(bias, kapa, gamma=2, init=0.25, size=size)
                grad_bias -= exp_deriv_l1(bias, beta=beta, gamma=2, init=0.25, size=size)

            return grad_input, grad_weight, grad_bias, None, None
    return _QuantLogDense
