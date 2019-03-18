import os
import sys
sys.path.insert(0, os.path.abspath('..'))


from models.FullNet.terMNIST import TerMNIST
import torch
import torch.utils.data
import numpy as np
import torchvision
# 0.9643

from device import device


BATCH_SIZE = 512
LEARNING_RATE = 0.0005
MOMENTUM = 0.6
EPOCH = 600
DATASET_SIZE = 60000
def adjust_learning_rate(optimizer, epoch):
    global LEARNING_RATE
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LEARNING_RATE * (1- 0.05 * (epoch))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

train_set = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/tmp/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=BATCH_SIZE, shuffle=True)




valid_set = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/tmp/mnist', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=BATCH_SIZE, shuffle=True)




model = TerMNIST(784, 10).to(device)
model.reset()



optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)#, momentum=MOMENTUM)

loss = torch.nn.CrossEntropyLoss().to(device)


for epoch in range(EPOCH):
    mean_loss = 0.0
    adjust_learning_rate(optimizer, epoch)
    for X,Y in train_set:

        # Reset des gradiants
        optimizer.zero_grad()

        # Mise en ligne des entrées pour le modèle
        X = X.view(-1,784)

        X = X.to(device)
        Y = Y.to(device)

        #Inférence
        outputs = model.forward(X,dropout=False)

        # Calcul du loss et propagation du gradient
        loss_val = loss(outputs, torch.autograd.Variable(Y.long()))
        mean_loss += loss_val.data


        loss_val.backward()

        # Mise à jour des poids
        optimizer.step()
        model.clamp()
    print("Epoch {}, mean_loss : {}, ".format(epoch, mean_loss/DATASET_SIZE))
    
    accucacy = 0.0
    for X, Y in valid_set:


        # Mise en ligne des entrées pour le modèle
        X = X.view(-1,784)

        X = X.to(device)
        Y = Y.to(device)

        #Inférence
        outputs = model.forward(X, dropout=True)

        _, val = torch.max(outputs, 1)
        accucacy += np.sum((val == Y.long()).data.cpu().numpy())
    print("\t Accuracy with drop out: {}".format(accucacy/10000))
    accucacy = 0.0
    for X, Y in valid_set:


        # Mise en ligne des entrées pour le modèle
        X = X.view(-1,784)

        X = X.to(device)
        Y = Y.to(device)

        #Inférence
        outputs = model.forward(X, dropout=False)

        _, val = torch.max(outputs, 1)
        accucacy += np.sum((val == Y.long()).data.cpu().numpy())
    print("\t Accuracy without drop out: {}".format(accucacy/10000))
