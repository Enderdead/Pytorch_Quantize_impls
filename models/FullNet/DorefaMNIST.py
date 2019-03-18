import torch 
import torch.nn as nn
from layers.dorefa_layers import LinearDorefa
from functions import dorefa_connect

class DorefaMNIST(torch.nn.Module):
    def __init__(self, in_features, out_features, num_units=2048, bitwight=3):
        super(DorefaMNIST, self).__init__()
        
        self.linear1 = LinearDorefa(in_features, num_units, bitwight=bitwight)
        self.norm1   = nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear2 = LinearDorefa(num_units, num_units, bitwight=bitwight)
        self.norm2  = nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear3 = LinearDorefa(num_units, num_units, bitwight=bitwight)
        self.norm3   = nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear4 = LinearDorefa(num_units, out_features,  bitwight=bitwight)
        self.norm4   = nn.BatchNorm1d(out_features, eps=1e-4, momentum=0.15)
        
        self.activation     = nn.ReLU()
        self.act_end = nn.LogSoftmax()
        self.bitwight = bitwight

    def reset(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()    
        self.linear3.reset_parameters()
        self.linear4.reset_parameters()


    def clamp(self):
        pass
        #self.linear1.clamp()
        #self.linear2.clamp()
        #self.linear3.clamp()
        #self.linear4.clamp()


    def forward(self, x, dropout=True):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = self.activation(self.linear1(x))
        x = self.norm1(x)
        x = dorefa_connect.DorefaQuant(x, self.bitwight)
        
        x = self.activation(self.linear2(x))
        x = self.norm2(x)
        x = dorefa_connect.DorefaQuant(x, self.bitwight)


        x = self.activation(self.linear3(x))
        x = self.norm3(x)
        x = dorefa_connect.DorefaQuant(x, self.bitwight)


        x = self.linear4(x)
        return self.act_end(x)
