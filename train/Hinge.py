import torch

class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)


def to_oneHot(tensor, nb_classes=10):

    one_hot = torch.FloatTensor(tensor.size(0), nb_classes)
    try:
        one_hot = one_hot.to(tensor.get_device())
    except RuntimeError:
        pass
    one_hot.zero_()
    one_hot.scatter_(1, tensor.view(tensor.size(0),1).long() , 1)
    return one_hot