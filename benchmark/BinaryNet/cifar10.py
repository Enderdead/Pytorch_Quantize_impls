import os
import sys
sys.path.insert(0, os.path.abspath('..'))


import QuantTorch as QtT 
#import torchvision.models as models
import torch
from models.cifar10.AlexNetBin import AlexNetBin
from torchbearer.callbacks import on_step_training
#from test import AlexNetBin
BATCH_SIZE = 256
EPOCH_NB = 54
DEVICE = QtT.device.device
init_lr = 0.1



alexnetBin = AlexNetBin(10).to("cpu")


train_set, valid_set, test_set = QtT.dataset.get_cifar10(BATCH_SIZE)

optimizer = torch.optim.Adam(alexnetBin.parameters(), lr=init_lr)
loss = torch.nn.CrossEntropyLoss()

from torchbearer import Trial, TEST_DATA
from torchbearer.callbacks import LambdaLR


regime =  { 0:  5e-3,
            40:  1e-3,
            80:  5e-4,
            100:  1e-4,
            120: 5e-5,
            140: 1e-5}

def lr_func(epoch):
    lr = 5e-3
    for key, value in regime.items():
        if key<=epoch:
            lr = value
        else:
            break
    print("set lr to : ", lr)
    return lr

scheduler = LambdaLR(lr_lambda=[lr_func])

@on_step_training
def clip(serial):
    alexnetBin.clip()

trial = Trial(alexnetBin, optimizer, criterion=loss, metrics=['acc', 'loss'], callbacks=[clip, scheduler]).to('cpu')
trial = trial.with_generators(train_generator=train_set, val_generator=valid_set, test_generator=test_set)
trial.run(epochs=150)

trial.evaluate(data_key=TEST_DATA)

alexnetBin.train()
torch.save(alexnetBin.state_dict(), open("./models/cifar10/alexnetBin.pth","wb"))