import torch
from common import front
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


def LogQuant(fsr=7, bitwight=3, lin_back=True, with_sign=False):
    class LogQuant_X(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if with_sign:
                return torch.sign(input)*torch.pow(torch.ones_like(input)*2, torch.clamp(torch.round(torch.log2(torch.abs(input))), fsr-2**bitwight ,fsr )) 

            return torch.pow(torch.ones_like(input)*2, torch.clamp(torch.round(torch.log2(torch.abs(input))), fsr-2**bitwight ,fsr )) 

        @staticmethod
        def backward(ctx, grad_output):
            if lin_back:
                grad_input = grad_output.clone()
                return grad_input

            return torch.pow(torch.ones_like(grad_output)*2, torch.clamp(torch.round(torch.log2(torch.abs(grad_output))), fsr-2**bitwight,fsr )) 

    return front(LogQuant_X)






FSR = 4
bitweight = 3
def LinQuant(fsr=7, bitwight=3):
    class LinQuant_X(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # Clip(Round(x/step)*step,0,2^(FSR)) with step = 2^{FSR-bitwight}
            step = torch.FloatTensor([2]).pow(FSR-bitwight).to(device)
            return torch.clamp(torch.round(input/step)*step, 0,2**FSR)  

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            step = torch.FloatTensor([2]).pow(FSR-bitwight).to(device)
            #return torch.clamp(torch.round(grad_output/step)*step, 0,torch.FloatTensor([2]).pow(FSR).to(device) ) 
            return grad_input

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
