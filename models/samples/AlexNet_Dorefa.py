import torch 
import torch.nn as nn
from layers.dorefa_layers import LinearDorefa, DorefaConv2d
from functions.dorefa_connect import nnDorefaQuant


class AlexNetDorefa(torch.nn.Module):
    def __init__(self, in_dim, out_dim, weight_bit=3, activation_bits=3):
        super(AlexNetDorefa, self).__init__()

        self.activation = nn.ReLu()
        self.quant_act  = nnDorefaQuant(bitwight=activation_bits)
        self.end_act    = torch.nn.LogSoftmax(dim=1)

        self.weight_bit = weight_bit
        self.activation_bits = activation_bits

        self.conv0 = DorefaConv2d(3, 64, kernel_size=3, stride=2, padding=1, weight_bit=self.weight_bit)
        self.norm0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=2)

        self.conv1 = DorefaConv2d(64, 192, kernel_size=3, padding=1, weight_bit=self.weight_bit)
        self.norm1 = nn.BatchNorm2d(192)
        self.pool1 = nn.MaxPool2d(kernel_size=2)


        self.conv2 = DorefaConv2d(192, 384, kernel_size=3, padding=1, weight_bit=self.weight_bit)
        self.norm2 = nn.BatchNorm2d(384)
        
        self.conv3 = DorefaConv2d(384, 256, kernel_size=3, padding=1, weight_bit=self.weight_bit)
        self.norm3 = nn.BatchNorm2d(256)
        
        self.conv4 = DorefaConv2d(256, 256, kernel_size=3, padding=1, weight_bit=self.weight_bit)
        self.norm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.lin5  = LinearDorefa(256 * 2 * 2, 4096, weight_bit=self.weight_bit)
        self.norm5  = nn.BatchNorm1d(4096)
    
        self.lin6  = LinearDorefa(4096, 4096, weight_bit=self.weight_bit)
        self.norm6  = nn.BatchNorm1d(4096)

        self.lin7 = LinearDorefa(4096, out_dim, weight_bit=self.weight_bit)


    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.activation(x)
        x = self.quant_act(x)
        x = self.pool0(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.quant_act(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.quant_act(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.quant_act(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.activation(x)
        x = self.quant_act(x)
        x = self.pool4(x)
        x = x.view(x.size(0), 256 * 2 * 2)

        x = self.lin5(x)
        x = self.norm5(x)
        x = self.activation(x)
        x = self.quant_act(x)

        x = self.lin6(x)
        x = self.norm6(x)
        x = self.activation(x)
        x = self.quant_act(x)

        x = self.lin7(x)
        x = self.end_act(x)
        return x
    

