import torch
from torch import nn
from layers.log_lin_layers import *
from functions.log_lin_connect import *

class VGGLinLogQuant(nn.Module):

    def __init__(self,dtype="lin", num_classes=10, Wbits=8, Abits=8):
        super(VGGLinLogQuant, self).__init__()

        self.out_act = nn.LogSoftmax(dim=1)
        self.act = nn.ReLU()
        self.quant_act = nnQuant(bitwight=Abits, fsr=1, with_sign=False)

        self.conv0 = QuantConv2d(3, 64, kernel_size=3, stride=1, padding=1, fsr=2, bitwight=Wbits, dtype=dtype)
        self.batch0 = torch.nn.BatchNorm2d(64)


        self.conv1 = QuantConv2d(64, 64, kernel_size=3, padding=1, fsr=2, bitwight=Wbits, dtype=dtype)
        self.batch1 = torch.nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = QuantConv2d(64, 128, kernel_size=3, padding=1, fsr=2, bitwight=Wbits, dtype=dtype)
        self.batch2 = torch.nn.BatchNorm2d(128)

        self.conv3 = QuantConv2d(128, 128, kernel_size=3, padding=1, fsr=2, bitwight=Wbits, dtype=dtype)
        self.batch3 = torch.nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)



        self.conv4 = QuantConv2d(128, 256, kernel_size=3, padding=1, fsr=2, bitwight=Wbits, dtype=dtype)
        self.batch4 = torch.nn.BatchNorm2d(256)

        self.conv5  = QuantConv2d(256, 256, kernel_size=3, padding=1, bitwight=Wbits,fsr=2, dtype=dtype)
        self.batch5 = torch.nn.BatchNorm2d(256)
        self.pool5  = nn.MaxPool2d(kernel_size=2)


        self.lin6   = LinearQuant(4096, 1024, fsr=1, bitwight=Wbits, dtype=dtype)
        self.batch6 = torch.nn.BatchNorm1d(1024)

        self.lin7 = LinearQuant(1024, 1024, fsr=1, bitwight=Wbits, dtype=dtype)
        self.batch7 = torch.nn.BatchNorm1d(1024)

        self.lin8 = LinearQuant(1024, 10, fsr=1, bitwight=Wbits, dtype=dtype)
        #self.lin7.weight.register_hook(print)

    def clamp(self):
        self.conv0.clamp()
        self.conv1.clamp()
        self.conv2.clamp()
        self.conv3.clamp()
        self.conv4.clamp()
        self.conv5.clamp()
        self.lin6.clamp()
        self.lin7.clamp()
        self.lin8.clamp()

    def forward(self, x):
        
        x = self.conv0(x)
        x = self.batch0(x)
        x = self.act(x)
        x = self.quant_act(x)
        #print(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act(x)
        x = self.quant_act(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.act(x)
        x = self.quant_act(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.act(x)
        x = self.quant_act(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = self.act(x)
        x = self.quant_act(x)

        x = self.conv5(x)
        x = self.batch5(x)
        x = self.act(x)
        x = self.quant_act(x)
        x = self.pool5(x)

        x = x.view(x.size(0), 4096)

        x = self.lin6(x)
        x = self.batch6(x)
        x = self.act(x)
        x = self.quant_act(x)

        x = self.lin7(x)
        x = self.batch7(x)
        x = self.act(x)
        x = self.quant_act(x)

        x= self.lin8(x)
        return self.out_act(x)

