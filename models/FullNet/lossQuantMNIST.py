import torch 
import torch.nn as nn
from layers.lossQuant_layers import LinearQuantLog, LinearQuantLin
from device import device

class QuantLossMNISTLin(torch.nn.Module):
    def __init__(self, in_features, out_features, num_units=2048, bottom=-1, top=1, size=5, alpha=0):

        super(QuantLossMNISTLin, self).__init__()
        
        self.linear1 = LinearQuantLin(in_features, num_units, bottom=bottom, top=top, size=size, alpha=alpha)
        self.norm1   =nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear2 = LinearQuantLin(num_units, num_units, bottom=bottom, top=top, size=size, alpha=alpha)
        self.norm2  =nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear3 = LinearQuantLin(num_units, num_units, bottom=bottom, top=top, size=size, alpha=alpha)
        self.norm3   =nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear4 = LinearQuantLin(num_units, out_features, bottom=bottom, top=top, size=size, alpha=alpha)
        self.norm4   =nn.BatchNorm1d(out_features, eps=1e-4, momentum=0.15)
        
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

    def set_alpha(self, alpha):
        self.linear1.alpha = torch.Tensor([alpha]).to(device)
        self.linear2.alpha = torch.Tensor([alpha]).to(device)
        self.linear3.alpha = torch.Tensor([alpha]).to(device)
        self.linear4.alpha = torch.Tensor([alpha]).to(device)
        

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = self.activation(self.linear1(x))
        x = self.norm1(x)
        
        x = self.activation(self.linear2(x))
        x = self.norm2(x)

        x = self.activation(self.linear3(x))
        x = self.norm3(x)

        x = self.linear4(x)
        return self.act_end(x)



class QuantLossMNISTLog(torch.nn.Module):
    def __init__(self, in_features, out_features, num_units=2048, gamma=2, init=0.25, size=5, alpha=0):

        super(QuantLossMNISTLog, self).__init__()
        
        self.linear1 = LinearQuantLog(in_features, num_units, gamma=gamma, init=init, size=size, alpha=alpha)
        self.norm1   =nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear2 = LinearQuantLog(num_units, num_units, gamma=gamma, init=init, size=size, alpha=alpha)
        self.norm2  =nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear3 = LinearQuantLog(num_units, num_units, gamma=gamma, init=init, size=size, alpha=alpha)
        self.norm3   =nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear4 = LinearQuantLog(num_units, out_features, gamma=gamma, init=init, size=size, alpha=alpha)
        self.norm4   =nn.BatchNorm1d(out_features, eps=1e-4, momentum=0.15)
        
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

    def set_alpha(self, alpha):
        self.linear1.alpha = torch.Tensor([alpha]).to(device)
        self.linear2.alpha = torch.Tensor([alpha]).to(device)
        self.linear3.alpha = torch.Tensor([alpha]).to(device)
        self.linear4.alpha = torch.Tensor([alpha]).to(device)
        

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = self.activation(self.linear1(x))
        x = self.norm1(x)
        
        x = self.activation(self.linear2(x))
        x = self.norm2(x)

        x = self.activation(self.linear3(x))
        x = self.norm3(x)

        x = self.linear4(x)
        return self.act_end(x)
