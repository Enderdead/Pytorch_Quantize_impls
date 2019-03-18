import torch
from functions.common import front
from device import device


class LogQuant(torch.autograd.Function):
    def __init(self, fsr=7, bitwight=3, with_sign=True, lin_back=True):
        super(LogQuant, self).__init__()
        self.fsr = fsr
        self.bitwight = bitwight
        self.with_sign = with_sign
        self.lin_back = lin_back

    def forward(self, ctx, input):
        if self.with_sign:
            return torch.sign(input)*torch.pow(torch.ones_like(input)*2, torch.clamp(torch.round(torch.log2(torch.abs(input))), self.fsr-2**self.bitwight ,self.fsr )) 
        return torch.pow(torch.ones_like(input)*2, torch.clamp(torch.round(torch.log2(torch.abs(input))), self.fsr-2**self.bitwight ,self.fsr )) 

    def backward(self, ctx, grad_output):
        if self.lin_back:
            grad_input = grad_output.clone()
            return grad_input
        return torch.pow(torch.ones_like(grad_output)*2, torch.clamp(torch.round(torch.log2(torch.abs(grad_output))), self.fsr-2**self.bitwight,self.fsr )) 


def LinQuant(fsr=7, bitwight=3):
    class LinQuant_temp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # Clip(Round(x/step)*step,0,2^(FSR)) with step = 2^{FSR-bitwight}
            step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
            return torch.clamp(torch.round(input/step)*step, 0,2**fsr)  

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
            #return torch.clamp(torch.round(grad_output/step)*step, 0,torch.FloatTensor([2]).pow(FSR).to(device) ) 
            return grad_input
    return LinQuant_temp


def nnLinQuant(fsr=7, bitwight=3):
    return front(LinQuant(fsr, bitwight))


def Quant(typ="lin", fsr=7, bitwight=3, with_sign=True, lin_back=True):
    if typ == "lin":
        return front(LinQuant(fsr, bitwight))
    else:
        return front(LogQuant(fsr=fsr, bitwight=bitwight, with_sign=with_sign, lin_back=lin_back))

"""

def LogQuant(fsr=7, bitwight=3, lin_back=True, with_sign=False):

    return front(LogQuant_X)


def LinQuant(fsr=7, bitwight=3):


    class fronteur(torch.nn.Module):
        def __init__(self):
            super(fronteur, self).__init__()

        def forward(self, x):
            return LinQuant_X.apply(x)

    return fronteur()



class LinearQuant(torch.nn.Linear):
    def __init__(self, *args, fsr=7, bitwight=3, lin_back=True, with_sign=True, **kwargs):
        super(LinearQuant, self).__init__(*args, **kwargs)
        self.quantizer = LogQuant(fsr, bitwight, lin_back, with_sign)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.quantizer(self.weight), None if self.bias is None else self.quantizer(self.bias))
"""