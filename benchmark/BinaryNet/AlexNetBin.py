import torch.nn as nn
from QuantTorch.layers import binary_layers as BL
from QuantTorch.functions import binary_connect as BC


NUM_CLASSES = 10


class AlexNetBin(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNetBin, self).__init__()
        self.features = nn.Sequential(
            BL.BinConv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            BC.BinaryConnect(),

            BL.BinConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),


      
            BC.BinaryConnect(),
            BL.BinConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),


            BC.BinaryConnect(),
            BL.BinConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),


            BC.BinaryConnect(),
            BL.BinConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),



            BL.BinConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

        )
        self.classifier = nn.Sequential(

            BC.BinaryConnect(),
            BL.LinearBin(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            BC.BinaryConnect(),

            BL.LinearBin(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            BC.BinaryConnect(),

            BL.LinearBin(1024, num_classes),
            nn.BatchNorm1d(10),
            nn.LogSoftmax()
        )

    def clip(self):
        for layer in self.features.modules():
            if isinstance(layer, (BL.BinConv2d, BL.LinearBin)):
                layer.clamp()
        for layer in self.classifier.modules():
            if isinstance(layer, (BL.BinConv2d, BL.LinearBin)):
                layer.clamp()



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 4 * 4)
        x = self.classifier(x)
        return x