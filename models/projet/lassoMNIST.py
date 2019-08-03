import torch 
import torch.nn as nn
from layers.elastic_layers import LinearQuantLin, LinearQuantLog
from functions import binary_connect
from utils.parameters import *

class Model(torch.nn.Module):

    static_params = [{'dtype':"lin", },{'dtype':"log", "gamma":2}]

    vars_params = [[DiscretParameter("size",2, 3, 4, 5),
                    DiscretParameter("num_units", 1024, 2048),
                    DiscretParameter("bottom", -1,-2),
                    DiscretParameter("top", 1,2),
                    UniformFloat("alpha_max", 0,2),
                    UniformFloat("beta_max", 0,2),
                    UniformInt("reg_start", 5,15)

                    ]
                    
                  ,[DiscretParameter("size",2, 3, 4, 5),
                    DiscretParameter("init",0.125,0.25,0.5),
                    DiscretParameter("num_units", 1024, 2048),
                    UniformFloat("alpha_max", 0,2),
                    UniformFloat("beta_max", 0,2),
                    UniformInt("reg_start", 5,15)
                   ]]
    
    opti_params = {
        "data": {"epoch" : 20},
        "optim": {"optimizer" : torch.optim.Adam},
     }

    var_opti_params = {
        "data":  [DiscretParameter("batch_size", 32, 64, 128, 256, 512),],
        "optim": [UniformLog("lr", 1e-5, 1),
                  UniformFloat("decay_lr", 0.0,0.2)]
    }

    MAX_EPOCH = 20

    def __init__(self, in_features=784, out_features=10, num_units=2048, dtype="lin", reg_start=5, alpha_max=2, beta_max=2,
                gamma=None, init=None, size=None,  # Log params
                bottom=None, top=None): # Lin params
        super(Model, self).__init__()

        if dtype=="lin":
            if bottom is None or top is None or size is None:
                raise TypeError("Lin mode need bottom, top and size kwargs")
            layers_args = {"bottom":bottom, "size":size, "top":top}
            layer = LinearQuantLin
        elif dtype=="log":
            if gamma is None or init is None or size is None:
                raise TypeError("Log mode need init, gamma and size kwargs")
            layers_args = {"init":init, "size":size, "gamma":gamma}
            layer = LinearQuantLog
        else:
            raise TypeError("Support only dtype in [\'lin\',\'log\']")
        

        self.alpha_max = alpha_max
        self.beta_max = beta_max
        self.reg_start = reg_start

        self.linear1 = layer(in_features, num_units, **layers_args)
        self.norm1   = nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear2 = layer(num_units, num_units, **layers_args)
        self.norm2  = nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear3 = layer(num_units, num_units, **layers_args)
        self.norm3   =nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15)

        self.linear4 = layer(num_units, out_features, **layers_args)
        self.norm4   =nn.BatchNorm1d(out_features, eps=1e-4, momentum=0.15)
        
        self.activation  = nn.ReLU()
        self.act_end     = nn.LogSoftmax(dim=1)

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

    def set_alpha(self, alpha):
        self.linear1.set_alpha(alpha)
        self.linear2.set_alpha(alpha)
        self.linear3.set_alpha(alpha)
        self.linear4.set_alpha(alpha)

    def set_beta(self, beta):
        self.linear1.set_alpha(beta)
        self.linear2.set_alpha(beta)
        self.linear3.set_alpha(beta)
        self.linear4.set_alpha(beta)


    def after_update(self,**kwargs):
        self.clamp()
        
    def after_epoch(self, epoch=0, **kwargs):
        if epoch<self.reg_start:
            self.set_alpha(0)
            self.set_beta(0)
        else:
            step_alpha = (self.alpha_max/(Model.MAX_EPOCH-self.reg_start))*(epoch-self.reg_start)
            step_beta = (self.beta_max/(Model.MAX_EPOCH-self.reg_start))*(epoch-self.reg_start)
            print("epoch {} : alpha {} , beta {}".format(epoch, step_alpha, step_beta))
            self.set_alpha(step_alpha)
            self.set_beta(step_beta)


    def forward(self, x, dropout=True):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.view(x.size(0), -1)
        x = self.activation(self.linear1(x))
        x = self.norm1(x)
        
        x = self.activation(self.linear2(x))
        x = self.norm2(x)

        x = self.activation(self.linear3(x))
        x = self.norm3(x)

        x = self.linear4(x)
        return self.act_end(x)
