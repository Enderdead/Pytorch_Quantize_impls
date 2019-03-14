import torch

def front(claaz):
    class fronteur(torch.nn.Module):
        def __init__(self):
            super(fronteur, self).__init__()
        def forward(self, x):
            return claaz.apply(x)
    return fronteur()