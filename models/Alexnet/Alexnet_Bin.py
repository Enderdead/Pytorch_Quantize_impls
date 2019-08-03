import torch
from torch import nn
from layers.binary_layers import *
from functions.binary_connect import *

class AlexNetBin(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetBin, self).__init__()
        coef = 3

        self.features = nn.Sequential(
                BinConv2d(3, 64*coef, kernel_size=11, stride=4, padding=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.BatchNorm2d(64*coef),
                nn.Hardtanh(inplace=True),
                BinaryConnect(stochastic=False),

                BinConv2d(64*coef, 192*coef, kernel_size=5, padding=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.BatchNorm2d(192*coef),
                nn.Hardtanh(inplace=True),

                BinaryConnect(stochastic=False),
                BinConv2d(192*coef, 384*coef, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(384*coef),
                nn.Hardtanh(inplace=True),

                BinaryConnect(stochastic=False),
                BinConv2d(384*coef, 256*coef, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(256*coef),
                nn.Hardtanh(inplace=True),

                BinaryConnect(stochastic=False),
                BinConv2d(256*coef, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.BatchNorm2d(256),
                nn.Hardtanh(inplace=True),
        )
        self.classifieur = nn.Sequential(
                BinaryConnect(stochastic=False),
                LinearBin(256 * 6 * 6, 4096),
                nn.BatchNorm1d(4096),
                nn.Hardtanh(inplace=True),

                BinaryConnect(stochastic=False),
                LinearBin(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.Hardtanh(inplace=True),
        
                BinaryConnect(stochastic=False),
                LinearBin(4096, 10),
                nn.LogSoftmax()
        )

    def clip(self):
        for layer in self.features.modules():
            if isinstance(layer, (BinConv2d, LinearBin)):
                layer.clamp()
        for layer in self.classifieur.modules():
            if isinstance(layer, (BinConv2d, LinearBin)):
                layer.clamp()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifieur(x)

        return x


if __name__ == "__main__":
    net = AlexNetBin()
