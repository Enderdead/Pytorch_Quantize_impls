import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import torch
from models.Alexnet.Alexnet_Bin import *
from models.Alexnet.Alexnet_BinTest import *
from torchvision import transforms
import torchvision


from device import device


BATCH_SIZE = 3

model = AlexNetBin().to(device)



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
eval_transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        

trainset = torchvision.datasets.CIFAR10(root='/tmp/CIFAR10', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/tmp/CIFAR10', train=False,
                                       download=True, transform=eval_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)



model.clip()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)# lr= 0.0001)#, momentum=0.8)

for epoch in range(400):  # loop over the dataset multiple times

    running_loss = 0.0
    correct_global = 0.0
    for inputs, labels in trainloader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model.forward(inputs)
        predicted = torch.argmax(outputs,dim=1)
        correct = (predicted == labels).sum().item()
        correct_global += correct
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        model.clip()
        # print statistics
        running_loss += loss.item()

        print(correct/BATCH_SIZE)
    """
    for inputs, labels in testloader:

        inputs = inputs.to(device)
        labels = labels.to(device)


        outputs = model.forward(inputs)
        predicted = torch.argmax(outputs,dim=1)


        correct += (predicted == labels).sum().item()

        testset.test_data.shape[0]
    """
    print('[%d] loss: %.5f accuracy: %.9f' %
            (epoch + 1, running_loss / trainset.train_data.shape[0], correct_global/ trainset.train_data.shape[0]))
    running_loss = 0.0


print('Finished Training')


