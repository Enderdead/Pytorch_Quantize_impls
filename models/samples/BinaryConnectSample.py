import torch 
import torch.nn as nn
from layers.binary_layers import LinearBin, BinarizeConv2d

class BinaryConnectCNN(torch.nn.Module):
    def __init__(self, in_features, out_features, num_units=2048):
        super(BinaryConnectCNN, self).__init__()  
        self.infl_ratio=1
        self.activation = nn.Hardtanh()
        self.out_act= nn.LogSoftmax(dim=1)        

        # C3 layer
        self.conv0 = BinarizeConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1)
        self.norm0 = nn.BatchNorm2d(128*self.infl_ratio)
        
        # C3 P2 layer
        self.conv1 = BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(128*self.infl_ratio)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # C2 palyer
        self.conv2 = BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(256*self.infl_ratio)

        #   C2 P2 layers
        self.conv3 = BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(256*self.infl_ratio)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv4 = BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(512*self.infl_ratio)
        
        self.conv5 = BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lin6 = LinearBin( 512 * 4 * 4, 1024)
        self.norm6 = nn.BatchNorm1d(1024)
            
        self.lin7 = LinearBin(1024, 1024)
        self.norm7 = nn.BatchNorm1d(1024)

        self.lin8 = LinearBin(1024, 10)
        self.norm8 = nn.BatchNorm1d(10, affine=False)


    def forward(self, x):
        x = x
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.activation(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)       

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.activation(x)       

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.activation(x)
        x = self.pool5(x)

        x = x.view(x.size(0),512 * 4 * 4)

        x = self.lin6(x)
        x = self.norm6(x)
        x = self.activation(x)

        x = self.lin7(x)
        x = self.norm7(x)
        x = self.activation(x)

        x = self.lin8(x)
        x=  self.norm8(x)
        return self.out_act(x)