import torch
from functions.common import front
from device import device

"""
Implement XNor primitive op:
https://arxiv.org/pdf/1603.05279.pdf
"""

DIM = 0
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
            input, weight, mean, bias = ctx.saved_variables
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
            input, weight, mean, bias = ctx.saved_variables

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

