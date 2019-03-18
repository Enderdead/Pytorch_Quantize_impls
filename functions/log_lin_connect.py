import torch
from functions.common import front
from device import device

def LogQuant(fsr=7, bitwight=3, with_sign=True, lin_back=True):
    class _LogQuant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if with_sign:
                return torch.sign(input)*torch.pow(torch.ones_like(input)*2, torch.clamp(torch.round(torch.log2(torch.abs(input))), fsr-2**bitwight ,fsr )) 
            return torch.pow(torch.ones_like(input)*2, torch.clamp(torch.round(torch.log2(torch.abs(input))), fsr-2**bitwight ,fsr )) 

        @staticmethod
        def backward(self, ctx, grad_output):
            if lin_back:
                grad_input = grad_output.clone()
                return grad_input
            return torch.sign(grad_output) * torch.pow(torch.ones_like(grad_output)*2, torch.clamp(torch.round(torch.log2(torch.abs(grad_output))), self.fsr-2**self.bitwight,self.fsr )) 
    return _LogQuant

def LinQuant(fsr=7, bitwight=3, with_sign=True, lin_back=True):
    class _LinQuant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # Clip(Round(x/step)*step,0,2^(FSR)) with step = 2^{FSR-bitwight}
            step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
            if with_sign:
                return torch.sign(input)*torch.clamp(torch.round(input/step)*step, 0,2**fsr)  
            return torch.clamp(torch.round(input/step)*step, 0,2**fsr)  

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
            if lin_back:
                return grad_input
            else:
                return torch.sign(grad_output)*torch.clamp(torch.round(grad_output/step)*step, 0,torch.FloatTensor([2]).pow(fsr).to(device) ) 
    return _LinQuant



def Quant(typ="lin", fsr=7, bitwight=3, with_sign=True, lin_back=True):
    if typ == "lin":
        return front(LinQuant(fsr=fsr, bitwight=bitwight, with_sign=with_sign, lin_back=lin_back))
    else:
        return front(LogQuant(fsr=fsr, bitwight=bitwight, with_sign=with_sign, lin_back=lin_back))

# TODO version log 
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
            input, weight, bias = ctx.saved_variables
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