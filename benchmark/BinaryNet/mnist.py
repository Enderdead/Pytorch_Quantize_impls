from torchbearer import Trial, TEST_DATA
from torchbearer.callbacks import LambdaLR
import QuantTorch as QtT 
import torch
from torchbearer.callbacks import on_step_training
from MLPBin import BinMNIST

BATCH_SIZE = 512
INIT_LEARNING_RATE = 0.1
EPOCH_NB = 600
DEVICE = QtT.device.device


alexnetBin = BinMNIST(784, 10).to(DEVICE)


train_set, valid_set, test_set = QtT.dataset.get_mnist(BATCH_SIZE)

optimizer = torch.optim.Adam(alexnetBin.parameters(), lr=INIT_LEARNING_RATE)
loss = torch.nn.CrossEntropyLoss()


regime =  { 0:  0.1,
            5:  0.05,
            10:  0.01,
            20:  1e-3,
            30: 5e-4,
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
    alexnetBin.clamp()

trial = Trial(alexnetBin, optimizer, criterion=loss, metrics=['acc', 'loss'], callbacks=[clip, scheduler]).to('cuda')
trial = trial.with_generators(train_generator=train_set, val_generator=valid_set, test_generator=test_set)
trial.run(epochs=EPOCH_NB)

trial.evaluate(data_key=TEST_DATA)

alexnetBin.train()
