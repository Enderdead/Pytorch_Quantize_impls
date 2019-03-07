import torch 
from binary_connect import *

class BinMNIST(torch.nn.Module):
    def __init__(self, num_units=2048):

        super(BinMNIST, self).__init__()
        self.linear1 = LinearBin(784, num_units)
        self.norm1   = torch.nn.BatchNorm1d(num_units)
        self.drop1   = torch.nn.Dropout(p=0.2)


        self.linear2 = LinearBin(num_units, num_units)
        self.norm2   = torch.nn.BatchNorm1d(num_units)
        self.drop2   = torch.nn.Dropout(p=0.2)

        self.linear3 = LinearBin(num_units, num_units)
        self.norm3   = torch.nn.BatchNorm1d(num_units)
        self.drop3   = torch.nn.Dropout(p=0.2)

        self.linear4 = LinearBin(num_units, 10)
        self.norm4   = torch.nn.BatchNorm1d(10)

        self.activation = torch.nn.LeakyReLU()

    def reset(self):
        self.linear1.init_weight()
        self.linear2.init_weight()    
        self.linear3.init_weight()
        self.linear4.init_weight()


    def clamp(self):
        self.linear1.weight.data.clamp_(min=-1, max=1)
        self.linear2.weight.data.clamp_(min=-1, max=1)
        self.linear3.weight.data.clamp_(min=-1, max=1)
        self.linear4.weight.data.clamp_(min=-1, max=1)


    def forward(self, x, dropout=True):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = self.linear1(x)
        x = self.activation(self.norm1(x))
        if dropout:
            x = self.drop1(x)
        
        x = self.linear2(x)
        x = self.activation(self.norm2(x))
        if dropout:
            x = self.drop2(x)

        x = self.linear3(x)
        x = self.activation(self.norm3(x))
        if dropout:
            x = self.drop3(x)

        x = self.linear4(x)
        x = self.norm4(x)
        return torch.nn.functional.softmax(x)
