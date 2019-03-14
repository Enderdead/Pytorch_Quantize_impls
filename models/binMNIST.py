import torch 
from layers import binary_layers

class BinMNIST(torch.nn.Module):
    def __init__(self, num_units=2048):

        super(BinMNIST, self).__init__()
        self.linear1 = binary_layers.LinearBin(784, num_units)
        self.norm1   = torch.nn.BatchNorm1d(num_units, eps=1e-05, momentum=0.15)
        self.drop1   = torch.nn.Dropout(p=0.2)


        self.linear2 = binary_layers.LinearBin(num_units, num_units)
        self.norm2   = torch.nn.BatchNorm1d(num_units, eps=1e-05, momentum=0.15)
        self.drop2   = torch.nn.Dropout(p=0.2)

        self.linear3 = binary_layers.LinearBin(num_units, num_units)
        self.norm3   = torch.nn.BatchNorm1d(num_units, eps=1e-05, momentum=0.15)
        self.drop3   = torch.nn.Dropout(p=0.2)

        self.linear4 = binary_layers.LinearBin(num_units, 10)
        self.norm4   = torch.nn.BatchNorm1d(10, eps=1e-05, momentum=0.15)

        self.activation = torch.nn.LeakyReLU()

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


    def forward(self, x, dropout=True):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = self.linear1(x)
        x = self.activation(self.norm1(x, eps=1e-05, momentum=0.15))
        if dropout:
            x = self.drop1(x)
        
        x = self.linear2(x)
        x = self.activation(self.norm2(x, eps=1e-05, momentum=0.15))
        if dropout:
            x = self.drop2(x)

        x = self.linear3(x)
        x = self.activation(self.norm3(x, eps=1e-05, momentum=0.15))
        if dropout:
            x = self.drop3(x)

        x = self.linear4(x)
        x = self.norm4(x, eps=1e-05, momentum=0.15)
        return torch.nn.functional.softmax(x)
