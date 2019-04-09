import torch
import numpy as np 

class Accuracy():
    def __init__(self, num_classes, one_hot=False):
        self.num_classes = num_classes
        self.one_hot = one_hot

    def __call__(self, outputs, labels):
        return self.forward(outputs, labels)

    def forward(self, outputs, labels):
        if self.one_hot==True:
            labels = torch.argmax(labels, 1)
        predictions = torch.argmax(outputs, 1)
        return np.sum((predictions == labels.long()).data.cpu().numpy())/(outputs.size(0))
