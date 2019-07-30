import torch
from log_lin_connect import *
from torch import nn

class LinearQuant(torch.nn.Linear):
    def __init__(self, *args, fsr=7, bitwight=3, lin_back=True, with_sign=True, **kwargs):
        super(LinearQuant, self).__init__(*args, **kwargs)
        self.quantizer = LogQuant(fsr, bitwight, lin_back, with_sign)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.quantizer(self.weight), None if self.bias is None else self.bias)


FSR = 7
BITWIGHT = 4
class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            LinearQuant(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            LinearQuant(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
