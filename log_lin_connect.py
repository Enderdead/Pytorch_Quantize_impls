import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

class LogQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.pow(torch.ones_like(input)*2, torch.round(torch.log2(torch.abs(input)))) 

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return torch.pow(torch.ones_like(grad_input)*2, torch.round(torch.log2(torch.abs(grad_input)))) 


FSR = 4
bitwight = 3

class LinQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Clip(Round(x/step)*step,0,2^(FSR)) with step = 2^{FSR-bitwight}
        step = torch.FloatTensor([2]).pow(FSR-bitwight).to(device)
        return torch.clamp(torch.round(input/step)*step, 0,torch.FloatTensor([2]).pow(FSR).to(device) )  )

    @staticmethod
    def backward(ctx, grad_output):
        step = torch.FloatTensor([2]).pow(FSR-bitwight).to(device)
        return torch.clamp(torch.round(grad_output/step)*step, 0,torch.FloatTensor([2]).pow(FSR).to(device) )  )
