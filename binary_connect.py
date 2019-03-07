import torch

class BinaryConnect(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)

        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.00001] = 0
        
        return grad_input


class LinearBin(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super(LinearBin, self).__init__(*args, **kwargs)

    def init_weight(self):
        torch.nn.init.uniform_(self.weight.data, -1, 1)

    def forward(self, input):
        return torch.nn.functional.linear(input, BinaryConnect.apply(self.weight), BinaryConnect.apply(self.bias))

