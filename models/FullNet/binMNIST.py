import torch 
import torch.nn as nn
from layers.binary_layers import LinearBin
from functions import binary_connect

class BinMNIST(torch.nn.Module):
    def __init__(self, in_features, out_features, num_units=2048):

        super(BinMNIST, self).__init__()
        
        self.linear1 = LinearBin(in_features, num_units)
        self.norm1   = nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear2 = LinearBin(num_units, num_units)
        self.norm2  = nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear3 = LinearBin(num_units, num_units)
        self.norm3   = nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear4 = LinearBin(num_units, out_features)
        self.norm4   = nn.BatchNorm1d(out_features, eps=1e-4, momentum=0.15)
        
        self.activation     = nn.ReLU()
        self.act_end = nn.LogSoftmax()

    def reset(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()    
        self.linear3.reset_parameters()
        self.linear4.reset_parameters()


    def clamp(self):
        self.linear1.clamp()
        self.linear2.clamp()
        self.linear3.clamp()
        self.linear4.clamp()


    def forward(self, x, dropout=True):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = self.activation(self.linear1(x))
        x = self.norm1(x)
        x = binary_connect.BinaryConnectDeterministic.apply(x)
        
        x = self.activation(self.linear2(x))
        x = self.norm2(x)
        x = binary_connect.BinaryConnectDeterministic.apply(x)


        x = self.activation(self.linear3(x))
        x = self.norm3(x)
        x = binary_connect.BinaryConnectDeterministic.apply(x)


        x = self.linear4(x)
        return self.act_end(x)
