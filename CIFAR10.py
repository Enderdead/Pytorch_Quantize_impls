import torch
from Alexnet import *
from torchvision import transforms
import torchvision


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


BATCH_SIZE = 128

model = AlexNet(10).to(device)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/tmp/CIFAR10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/tmp/CIFAR10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)



criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#, momentum=0.8) 0.0001

for epoch in range(40):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0.0
    for inputs, labels in trainloader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model.forward(inputs)
        predicted = torch.argmax(outputs,dim=1)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
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
            (epoch + 1, running_loss / trainset.train_data.shape[0], correct/ trainset.train_data.shape[0]))
    running_loss = 0.0


print('Finished Training')




