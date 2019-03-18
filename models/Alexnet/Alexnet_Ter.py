import torch
from torch import nn
from layers.binary_layers import *
from functions.terner_connect import *
class AlexNetBin(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetBin, self).__init__()
        self.features = nn.Sequential(
            Conv2dBin(3, 64, kernel_size=3, stride=2, padding=1),
            BinaryConnect(),
            nn.MaxPool2d(kernel_size=2),
            Conv2dBin(64, 192, kernel_size=3, padding=1),
            BinaryConnect(),
            nn.MaxPool2d(kernel_size=2),
            Conv2dBin(192, 384, kernel_size=3, padding=1),
            BinaryConnect(),
            Conv2dBin(384, 256, kernel_size=3, padding=1),
            BinaryConnect(),
            Conv2dBin(256, 256, kernel_size=3, padding=1),
            BinaryConnect(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            #nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
