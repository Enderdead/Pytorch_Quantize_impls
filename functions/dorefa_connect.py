import torch
from fonctions.common import front

def _quantize(x, bits=3):
    two = torch.ones_like(x)*2
    return ((1)/(torch.pow(two,bits)-1))*torch.round((torch.pow(two,bits)-1)*x)

def nnDorefaQuant(bitwight=3):
    class _Quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return _quantize(input, bits=bitwight)
        def backward(ctx, grad_ouput):
            return grad_ouput.clone()
    return front(_Quant)
                

def DorefaQuant(x, bitwight=3):
    class _Quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return _quantize(input, bits=bitwight)
        def backward(ctx, grad_ouput):
            return grad_ouput.clone()
    return _Quant.apply(x)


def QuantDense(input, weight, bias=None, bitwight=3):
    class _QuantDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            max_weight = torch.max(torch.abs(weight))
            weight_q = 2*_quantize(0.5  + torch.nn.functional.tanh(weight)/(2*torch.nn.functional.tanh(max_weight))   , bits=bitwight) - 1
            ctx.save_for_backward(input, weight, weight_q, max_weight, bias)
            output = torch.nn.functional.linear(input, weight_q, bias)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, weight_q, max_weight, bias = ctx.saved_variables
            grad_input = grad_weight = grad_bias = None
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight_q)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
                grad_weight =  grad_weight * (1 - torch.pow(weight,2)) /  torch.nn.functional.tanh(max_weight) 
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
            return grad_input, grad_weight, grad_bias

    return _QuantDense.apply(input, weight, bias)


def QuantConv2d(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1, bitwight=3):
    class _QuantConv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            max_weight = torch.max(torch.abs(weight))
            weight_q = 2*_quantize(0.5  + torch.nn.functional.tanh(weight)/(2*torch.nn.functional.tanh(max_weight))   , bits=bitwight) - 1
            ctx.save_for_backward(input, weight, weight_q, max_weight, bias)
            output = torch.nn.functional.conv2d(input, weight_q, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, weight_q, max_weight, bias = ctx.saved_variables
            
            weight_q = torch.sign(weight)*torch.pow(torch.ones_like(weight)*2, torch.clamp(torch.round(torch.log2(torch.abs(weight))), fsr-2**bitwight ,fsr )) 

            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(input.size(), weight_q, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
                grad_weight = grad_weight * (1 - torch.pow(weight,2)) /  torch.nn.functional.tanh(max_weight) 
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0).sum(1).squeeze(1).sum(-1).squeeze(-1)

            if bias is not None:
                return grad_input, grad_weight, grad_bias
            else:
                return grad_input, grad_weight  

    return _QuantConv2d.apply(input, weight, bias)

