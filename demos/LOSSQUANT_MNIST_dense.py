import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from models.FullNet.lossQuantMNIST import QuantLossMNIST
import torch
import torch.utils.data
import numpy as np
import torchvision
from progress.bar import ChargingBar
from pickle import dump
from device import device
def save_weight(model, path):
  data = np.concatenate([model.linear1.weight.cpu().detach().numpy().reshape(-1), model.linear2.weight.cpu().detach().numpy().reshape(-1), model.linear3.weight.cpu().detach().numpy().reshape(-1), model.linear4.weight.cpu().detach().numpy().reshape(-1)],axis=0)
  dump(data, open(path, "wb"))



BATCH_SIZE = 200
LEARNING_RATE = 0.001
BOTTOM = -1
TOP = 1
ALPHA = [0.0, 0.0, 0.01, 0.01, 0.5, 0.5, 1]
SIZE = 5
EPOCH = 7
DATASET_SIZE = 60000

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




model = QuantLossMNIST(784, 10, top=TOP, bottom=BOTTOM, size=SIZE, alpha=ALPHA[0]).to(device)
model.reset()



optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)#, momentum=MOMENTUM)

loss = torch.nn.CrossEntropyLoss().to(device)


for epoch in range(EPOCH):
    optimizer.zero_grad()
    model.train()
    mean_loss = 0.0
    print(" Set alpha to -> ",ALPHA[epoch] )
    bar = ChargingBar('Learning', max=1+(DATASET_SIZE//BATCH_SIZE))
    model.set_alpha(ALPHA[epoch])
    for X,Y in train_set:
        bar.next()
        # Reset des gradiants
        optimizer.zero_grad()

        # Mise en ligne des entrées pour le modèle
        X = X.view(-1,784)

        X = X.to(device)
        Y = Y.to(device)

        #Inférence
        outputs = model.forward(X)

        # Calcul du loss et propagation du gradient
        loss_val = loss(outputs, torch.autograd.Variable(Y.long()))
        mean_loss += loss_val.data


        loss_val.backward()

        # Mise à jour des poids
        optimizer.step()
        model.clamp()
    print("\nEpoch {}, mean_loss : {}, ".format(epoch, mean_loss/DATASET_SIZE))
    print(model.linear1.weight)
    model.eval()
    print(model.linear1.weight)

    accucacy = 0.0
    for X, Y in valid_set:


        # Mise en ligne des entrées pour le modèle
        X = X.view(-1,784)

        X = X.to(device)
        Y = Y.to(device)

        #Inférence
        outputs = model.forward(X)

        _, val = torch.max(outputs, 1)
        accucacy += np.sum((val == Y.long()).data.cpu().numpy())
    print("\t Accuracy : {}".format(accucacy/10000))
    save_weight(model, "save_weight_"+str(epoch)+".piclke")
