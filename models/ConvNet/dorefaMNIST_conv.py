import torch 
import torch.nn as nn
from layers.dorefa_layers import LinearDorefa, DorefaConv2d
from functions import dorefa_connect


class DorefaMNIST(torch.nn.Module):
    def __init__(self, in_features, out_features, num_units=2048, bitwight=3):
        super(DorefaMNIST, self).__init__()

        self.conv1 = DorefaConv2d(1, 32, kernel_size=3, padding=1, bitwight=bitwight)
        self.norm1   = nn.BatchNorm2d(32, eps=1e-4, momentum=0.15)

        self.conv2 = DorefaConv2d(32, 64, kernel_size=3, padding=1, bitwight=bitwight)
        self.pool2   =  nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2   = nn.BatchNorm2d(64, eps=1e-4, momentum=0.15)

        self.linear3 = LinearDorefa(10816, num_units, bitwight=bitwight)#64 * 13 *  13
        self.norm3   = nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear4 = LinearDorefa(num_units, out_features, bitwight=bitwight)
        self.norm4   = nn.BatchNorm1d(out_features, eps=1e-4, momentum=0.15)
        
        self.activation     = nn.ReLU()
        self.act_end = nn.LogSoftmax()
        self.bitwight = bitwight

    def reset(self):
        pass
        #self.conv1.reset_parameters()
        #self.conv2.reset_parameters()    
        #self.linear3.reset_parameters()
        #self.linear4.reset_parameters()


    def clamp(self):
        pass
        #self.conv1.clamp()
        #self.conv2.clamp()
        #self.linear3.clamp()
        #self.linear4.clamp()


    def forward(self, x, dropout=True):
        x = self.activation(self.conv1(x.view(-1, 1, 28, 28)))
        x = self.norm1(x)
        x = dorefa_connect.DorefaQuant(x, self.bitwight)
        
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)
        x = dorefa_connect.DorefaQuant(x, self.bitwight)

        x = x.view(x.size(0), -1)
        x = self.activation(self.linear3(x))
        x = self.norm3(x)
        x = dorefa_connect.DorefaQuant(x, self.bitwight)

        x = self.linear4(x)
        return x
