import torch


def safeSign(tensor):
    result = torch.sign(tensor)
    result[result==0] = 1
    return result


def front(claaz):
    """
    Return a Module proxy of your claaz. 
    """
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