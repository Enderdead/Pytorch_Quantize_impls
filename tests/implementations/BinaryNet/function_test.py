from QuantTorch.functions import binary_connect
from .BinaryNet_pytorch.models import binarized_modules
import torch
from math import copysign

def sign(x):
    return copysign(1, x)


def test_with_ref_det():
    x_1 = torch.Tensor([1,0.5,-0.33])
    x_2 = torch.Tensor([[1,0.5,-0.33],[0.,0.1,-0.8]])

    assert torch.all(torch.eq(binarized_modules.Binarize(x_1,quant_mode='det'),
                    binary_connect.BinaryConnectDeterministic.apply(x_1)))

    assert torch.all(torch.eq(binarized_modules.Binarize(x_2,quant_mode='det'),
                    binary_connect.BinaryConnectDeterministic.apply(x_2)))


def test_det_apply():
    x_1 = torch.Tensor([1,0.5,-0.33])
    x_2 = torch.Tensor([[4,2,-4],[0.,0.1,-0.8]])

    y_1 = torch.sign(x_1)
    y_2 = torch.sign(x_2)
    assert torch.all(torch.eq(y_1,
                    binary_connect.BinaryConnectDeterministic.apply(x_1)))

    assert torch.all(torch.eq(y_2,
                    binary_connect.BinaryConnectDeterministic.apply(x_2)))


def test_sto_apply():
    weight = torch.Tensor([1,0,-1]).view(1,-1)
    results = binary_connect.BinaryConnectStochastic.apply(weight.view(1,-1))
    for _ in range(150):
        results = torch.cat( (results, binary_connect.BinaryConnectStochastic.apply(weight)),0)
    
    assert torch.mean(results,0)[0] == 1
    assert torch.mean(results,0)[2] == -1
    assert -0.2 < torch.mean(results,0)[1] < 0.2




def test_det_backprob():
    #Setup input
    inputs = torch.autograd.Variable(torch.Tensor([2,-0.5,1]).view(1,-1), requires_grad=True)


    #First test simple backprob and forward case
    weight_1 = torch.autograd.Variable(torch.Tensor([0.5,-0.5,0.5]).view(-1,1), requires_grad=True)

    # loss = inputs*BIN(weight)
    loss_1 = torch.mm(inputs,binary_connect.BinaryConnectDeterministic.apply(weight_1))

    assert torch.all(torch.eq(loss_1, torch.Tensor([sign(0.5)*2 + sign(-0.5)*-0.5 + sign(0.5)*1])))
    loss_1.backward()

    # weight.grad = d(inputs*BIN(weight))/d(BIN(weight))    *    d(BIN(weight)) / d(weight)   <= (equals to identity cf paper)
    assert torch.all(torch.eq(weight_1.grad, inputs.view(weight_1.grad.shape)))

    weight_1.grad.data.zero_()

    # Try with more complex backprob 
    square_bin_weight1 = torch.pow(binary_connect.BinaryConnectDeterministic.apply(weight_1), 2)

    loss_2 = torch.mm(inputs,square_bin_weight1)
    loss_2.backward()

    # weight.grad = d(inputs*BIN(weight)²)/d(BIN(weight))    *    d(BIN(weight)) / d(weight)   <= (equals to identity cf paper)
    assert torch.all(torch.eq(weight_1.grad, 2*inputs.view(weight_1.grad.shape)*binary_connect.BinaryConnectDeterministic.apply(weight_1)))


    # Test if grad is zero when weight's magnitude is too hight 

    weight_2 = torch.autograd.Variable(torch.Tensor([2,0.5,0.5]).view(-1,1), requires_grad=True)
    
    loss_3 = torch.mm(inputs, binary_connect.BinaryConnectDeterministic.apply(weight_2))
    loss_3.backward()
    assert torch.all(torch.eq(weight_2.grad, torch.Tensor([0,-0.5,1]).view(-1,1)))




def test_sto_backprob():
    #Setup input
    inputs = torch.autograd.Variable(torch.Tensor([2,-0.5,1]).view(1,-1), requires_grad=True)


    #First test simple backprob and forward case
    weight_1 = torch.autograd.Variable(torch.Tensor([0.5,-0.5,0.5]).view(-1,1), requires_grad=True)
    weight_1_quant = binary_connect.BinaryConnectStochastic.apply(weight_1)
    # loss = inputs*BIN(weight)
    loss_1 = torch.mm(inputs,weight_1_quant)
    assert torch.all(torch.eq(loss_1, torch.Tensor([weight_1_quant[0]*2 + weight_1_quant[1]*-0.5 + weight_1_quant[2]*1])))
    loss_1.backward()

    # weight.grad = d(inputs*BIN(weight))/d(BIN(weight))    *    d(BIN(weight)) / d(weight)   <= (equals to identity cf paper)
    assert torch.all(torch.eq(weight_1.grad, inputs.view(weight_1.grad.shape)))


    weight_2 = torch.autograd.Variable(torch.Tensor([0.5,-0.5,0.5]).view(-1,1), requires_grad=True)
    weight_2_quant = binary_connect.BinaryConnectStochastic.apply(weight_2)


    # Try with more complex backprob 
    square_bin_weight2 = torch.pow(weight_2_quant, 2)

    loss_2 = torch.mm(inputs,square_bin_weight2)
    loss_2.backward()

    # weight.grad = d(inputs*BIN(weight)²)/d(BIN(weight))    *    d(BIN(weight)) / d(weight)   <= (equals to identity cf paper)
    assert torch.all(torch.eq(weight_2.grad, 2*inputs.view(weight_2.grad.shape)*weight_2_quant))


    # Test if grad is zero when weight's magnitude is too hight 

    weight_3 = torch.autograd.Variable(torch.Tensor([2,0.5,0.5]).view(-1,1), requires_grad=True)
    
    loss_3 = torch.mm(inputs, binary_connect.BinaryConnectStochastic.apply(weight_3))
    loss_3.backward()
    assert weight_3.grad[0] == 0.0


