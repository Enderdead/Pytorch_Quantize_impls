import torch


def front(claaz):
    class fronteur(torch.nn.Module):
        def __init__(self):
            super(fronteur, self).__init__()
        def forward(self, x):
            return claaz.apply(x)

    return fronteur()


def front2(claaz):
    class fronteur(torch.nn.Module):
        def __init__(self):
            super(fronteur, self).__init__()
            self.core = claaz

        def forward(self, x):
            return self.core.apply(x)
    return fronteur()