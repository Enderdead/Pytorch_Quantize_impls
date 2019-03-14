import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


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

        return grad_input


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





