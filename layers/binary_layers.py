import torch 
from functions import binary_connect



class LinearBin(torch.nn.Linear):
    def __init__(self, *args, stochastic=False, **kwargs):
        super(LinearBin, self).__init__(*args, **kwargs)
        self.stochastic = stochastic

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, -1, 1)
        if not self.bias is None:
            torch.nn.init.uniform_(self.bias, -1 , 1)

    def clamp(self):
        self.weight.data.clamp_(-1, 1)
        if not self.bias is None:
            self.bias.data.clamp_(-1, 1)

    def forward(self, input):
        return torch.nn.functional.linear(input, binary_connect.BinaryConnectStochastic.apply(self.weight), None if self.bias is None else binary_connect.BinaryConnectStochastic.apply(self.bias))

