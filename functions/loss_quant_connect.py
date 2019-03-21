import torch
from functions.common import front
from device import device

def lin_deriv(x, alpha, top=1,  bottom=-1, size=5):
    delta = (top-bottom)/(size-1)
    res = torch.zeros_like(x)

    for i in range(size):
        res += (-alpha*x +(bottom+i*delta)*alpha)*(x<(bottom+(i)*delta+delta/2)).float()*(x>(bottom+(i)*delta-delta/2)).float()
    return res



def QuantDense(size=5, bottom=-1, top=1):
    class _QuantDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias, alpha):
            alpha.to(device)
            ctx.save_for_backward(input, weight, bias, alpha)
            return torch.nn.functional.linear(input, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, alpha = ctx.saved_variables
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)

            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
                grad_weight -= lin_deriv(weight, alpha=alpha, bottom=bottom, top=top)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
                grad_bias -= lin_deriv(bias, alpha=alpha, bottom=bottom, top=top)

            return grad_input, grad_weight, grad_bias, None
    return _QuantDense




def QuantConv2d(size=5, bottom=-1, top=1, stride=1, padding=1, dilation=1, groups=1):
    class _QuantConv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias, alpha):
            ctx.save_for_backward(input, weight, bias, alpha)
            output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, alpha = ctx.saved_variables

            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(input.size(), weight, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
                grad_weight -= lin_deriv(weight, alpha, top, bottom, size)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0).sum(1).squeeze(1).sum(-1).squeeze(-1)
                grad_bias -= lin_deriv(bias, alpha, top, bottom, size)

            return grad_input, grad_weight, grad_bias, None

    return _QuantConv2d