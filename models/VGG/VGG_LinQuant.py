import torch
from torch import nn
from layers.linquant_layers import *
from functions.log_lin_connect import *

class VGGLinQuant(nn.Module):

    def __init__(self, num_classes=10, bitwight=8):
        super(VGGLinQuant, self).__init__()
        cons = 7
        self.features = nn.Sequential(
            QuantLinConv2d(3, 64, kernel_size=3, stride=1, padding=1, fsr=7, bitwight=bitwight),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(),
            Quant(bitwight=bitwight, fsr=2, with_sign=False),


            QuantLinConv2d(64, 64, kernel_size=3, padding=1, fsr=7, bitwight=bitwight),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Quant(bitwight=bitwight, fsr=2, with_sign=False),
            nn.MaxPool2d(kernel_size=2),

            QuantLinConv2d(64, 128, kernel_size=3, padding=1, fsr=7, bitwight=bitwight),
            torch.nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Quant(bitwight=bitwight, fsr=2, with_sign=False),

            QuantLinConv2d(128, 128, kernel_size=3, padding=1, fsr=7, bitwight=bitwight),
            torch.nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Quant(bitwight=bitwight, fsr=2, with_sign=False),
            nn.MaxPool2d(kernel_size=2),



            QuantLinConv2d(128, 256, kernel_size=3, padding=1, fsr=7, bitwight=bitwight),
            torch.nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Quant(bitwight=bitwight, fsr=2, with_sign=False),

            QuantLinConv2d(256, 256, kernel_size=3, padding=1, bitwight=bitwight),
            torch.nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Quant(bitwight=bitwight, fsr=2, with_sign=False),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            LinearLinQuant(4096, 1024, fsr=7, bitwight=bitwight),
            torch.nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            Quant(bitwight=bitwight, fsr=2, with_sign=False),

            LinearLinQuant(1024, 1024, fsr=7, bitwight=bitwight),
            torch.nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            Quant(bitwight=bitwight, fsr=2, with_sign=False),

            LinearLinQuant(1024, 10, fsr=7, bitwight=bitwight),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x

