import torch
from fonctions.common import front

from device import device


class TernerConnectDeterministic(torch.autograd.Function):
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


class TernerConnectStochastic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        sign = torch.sign(input)
        z = torch.rand_like(input, requires_grad=False)
        return sign - (z>torch.abs(input)).type(torch.FloatTensor).to(device) 

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_input



def TernerConnect(stochastic=False):
    act = TernerConnectStochastic if stochastic else TernerConnectDeterministic
    return front(act)


