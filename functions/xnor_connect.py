import torch
from functions.common import front
from device import device
import warnings
warnings.simplefilter("always",DeprecationWarning)
warnings.warn("Module not finished due to  gradient implementation on conv layer !", ImportWarning)

"""
Implement XNor primitive op:
https://arxiv.org/pdf/1603.05279.pdf
"""

DIM = 0



def _quantOpXnor(dim=1):
    class _QuantXNOR(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            mean = torch.mean(input) if dim<0 else torch.mean(input, dim)
            ctx.save_for_backward(input, mean)
            
            if dim<0:
                return torch.sign(input)*mean
            else:
                form_mean = {0 : (1,-1), 1 : (-1,1)}[dim]
                return torch.sign(input)*mean.view(form_mean)
        @staticmethod
        def backward(ctx, grad_outputs):
            input, mean = ctx.saved_tensors
            sgn_input = torch.sign(input)
            if dim<0:
                return sgn_input*torch.mean(grad_outputs*sgn_input) + grad_outputs*mean
            form_mean = {0 : (1,-1), 1 : (-1,1)}[dim]

            return sgn_input*torch.mean( grad_outputs*sgn_input, dim ,keepdim=True) + grad_outputs*mean.view(form_mean).expand(input.size())
    return _QuantXNOR

def nnQuantXnor(dim=1):
    if not dim in [-1, 0, 1]:
        raise RuntimeError(" Please use a correct dim between -1, 0, 1")
    """
    Apply a Xnor binarizarion on classic input with 2 dimenssion (on full connected env).
    output = sign(input)*mean(|input|)
    param: 
        dim :  1 compute mean along 2nd dim (default behaviour)
        dim :  0 compute mean along 1st dim (along each dim of batch's examples)
        dim : -1 compute mean along all data.
    """
    op = _quantOpXnor(dim)
    return front(op)

def QuantXnor(input, dim=1):
    if not dim in [-1, 0, 1]:
        raise RuntimeError(" Please use a correct dim between -1, 0, 1")
    """
    Apply a Xnor binarizarion on classic input with 2 dimenssion (on full connected env).
    output = sign(input)*mean(|input|)
    param: 
        dim :  1 compute mean along 2nd dim (default behaviour)
        dim :  0 compute mean along 1st dim (along each dim of batch's examples)
        dim : -1 compute mean along all data.
    """
    op = _quantOpXnor(dim)
    return op.apply(input)


def _quantOpXnor2d(kernel_size, stride=1, padding=1, dilation=1, groups=1, form="NCHW"):
    if not form in ["NHWC", "NCHW"]:
        raise RuntimeError("Input form insupported ")

    if type(kernel_size) !=int:
        raise RuntimeError("Only int kernel_size supported (square kernel)")

    class _QuantXNOR2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            input_mean_channel = torch.mean(input, 1, keepdim=True)
            kernel = torch.ones(1, 1,kernel_size, kernel_size).to(input.device)
            kernel.data.mul_(1/(kernel_size**2))
            input_mean = torch.nn.functional.conv2d(input_mean_channel,kernel ,bias=False,stride=1, padding=1, dilation=1, groups=1)
            input_mean.require_grad = False
            ctx.save_for_backward(input, input_mean)
            return torch.sign(input)*input_mean

        @staticmethod
        def backward(ctx, grad_outputs):
            raise NotImplementedError("Conv XNor net not implemented !")
                


def XNORDense(dim=[0,1]):
    """
        Return a XNOR Dense op with backprob. Apply a binarization on Weight only with a reduct mean.
        Wi_b = sign(W)*mean(|Wi|) with mean compute along dim given.
        
    """
    class _XNORDense(torch.autograd.Function):
        """
        Applies a linear transformation to the incoming data: y=W_b.x+b
        With W_b a binarized transformation of W with a deterministic way.
        Wi_b = sign(W)*mean(|Wi|)

        Apply back prob with real grad : 
        
        dC/dWi = n^{-1}*sign(Wi)*sum(dC/dWj_b * sign(Wj) ) + dC/dWi_b * alpha
        with alpha = mean(|W|)
        """
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            mean = torch.mean(torch.abs(weight), DIM, keepdim=True)
            weight_q = torch.sign(weight)*mean
            ctx.save_for_backward(input, weight, mean, bias)
            output = torch.nn.functional.linear(input, weight_q, bias)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, mean, bias = ctx.saved_tensors
            weight_q = torch.sign(weight)*mean
            grad_input = grad_weight = grad_bias = None
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight_q)
            if ctx.needs_input_grad[1]:
                grad_weight_temp = grad_output.t().mm(input)
                grad_weight = mean*grad_weight_temp +torch.sign(weight) * torch.mean(grad_weight_temp*torch.sign(weight) , DIM, keepdim=True)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
            return grad_input, grad_weight, grad_bias
    
    return _XNORDense


def XNORConv2d(dim=[0,1], quant_input=False,  stride=1, padding=1, dilation=1, groups=1):
    # TODO backprob input quant
    class _XNORConv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            mean_weight = torch.mean(torch.abs(weight), dim, keepdim=True)
            weight_b = torch.sign(weight)*mean_weight
            if quant_input:
                input = torch.sign(input) *torch.mean(torch.abs(input), 1, keepdim=True)
            ctx.save_for_backward(input, weight, mean_weight, bias)
            output = torch.nn.functional.conv2d(input, weight_b, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, mean, bias = ctx.saved_tensors

            weight_b = torch.sign(weight)*mean
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(input.size(), weight_b, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if ctx.needs_input_grad[1]:
                grad_weight_temp = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
                grad_weight = mean*grad_weight_temp +torch.sign(weight) * torch.mean(grad_weight_temp*torch.sign(weight) , DIM, keepdim=True)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0).sum(1).squeeze(1).sum(-1).squeeze(-1)

            if bias is not None:
                return grad_input, grad_weight, grad_bias
            else:
                return grad_input, grad_weight  

    return _XNORConv2d

