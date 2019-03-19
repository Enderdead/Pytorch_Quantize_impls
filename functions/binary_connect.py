import torch
from functions.common import front
from device import device

class BinaryDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        weight_b = torch.sign(weight)
        output = torch.nn.functional.linear(input, weight_b, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        weight_b = torch.sign(weight)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_b)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias



def BinaryConv2d(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
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

    return _BinaryConv2d.apply(input, weight, bias)


class BinaryConnectDeterministic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_output



class BinaryConnectStochastic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        z = torch.rand_like(input, requires_grad=False)
        p = ((torch.clamp(input, -1, 1) + 1) / 2).pow(2)

        return -1.0 + 2.0 * (z<p).type(torch.FloatTensor).to(device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_input

# AP2 = sign(x) × 2round(log2jxj)
def AP2(x):
    two = torch.ones_like(x)*2
    return torch.sign(x) * torch.pow(two,torch.round(torch.log2(torch.abs(x))))


class ShiftBatch1d(torch.autograd.Function):
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
            grad_input = grad_output*weight
            grad_input = grad_input/sqrtvar

        if ctx.needs_input_grad[1]:
            grad_running_mean = None#torch.zeros_like(weight)

        if ctx.needs_input_grad[2]:
            grad_running_var = None #torch.zeros_like(weight)

        if ctx.needs_input_grad[3]:
            grad_weight = grad_output*norm_inputs

        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0).squeeze(0)

        if ctx.needs_input_grad[5]:
            grad_eps = None#torch.zeros_like(eps)
        
        return grad_input, grad_running_mean, grad_running_var, grad_weight, grad_bias,  grad_eps






def BinaryConnect(stochastic=False):
    act = BinaryConnectStochastic if stochastic else BinaryConnectDeterministic
    return front(act)



