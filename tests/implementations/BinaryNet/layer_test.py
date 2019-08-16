from QuantTorch.layers import LinearBin, BinConv2d
from QuantTorch.functions import BinaryConnectDeterministic
import torch
from torch.autograd import Variable
from math import copysign

def sign(x):
    return copysign(1, x)

def test_lin_layer_forward():
    """
    Test if forward pass works fine.
    """

    # Setup layer and input
    lin = LinearBin(3,1, bias=False)
    lin.weight.data.copy_(torch.Tensor([0.5,0,-0.5]).view(1,3))
    inputs = Variable(torch.Tensor([[2,1,-3],]))

    #  check the forward pass 
    assert torch.all(torch.eq(lin(inputs), torch.mm(torch.Tensor([[2,1,-3],]),torch.Tensor([sign(0.5),sign(0.1),sign(-0.5)]).view(3,1))))
    
    # check with a bias
    lin2 = LinearBin(3,1, bias=True)
    lin2.weight.data.copy_(torch.Tensor([0.5,0,-0.5]).view(1,3))
    lin2.bias.data.copy_(torch.Tensor([3]).view(1))
    inputs = Variable(torch.Tensor([[2,1,-3],]))

    assert torch.all(torch.eq(lin(inputs), torch.mm(torch.Tensor([[2,1,-3],]),3+torch.Tensor([sign(0.5),sign(0.1),sign(-0.5)]).view(3,1))))
    

def test_conv_layer_forward():
    """
    Check the forward pass with conv layer
    """
    #Setup layer and inputs
    conv = BinConv2d(2,1, [2,2],stride=1, bias=False)
    conv.weight.data.copy_(torch.Tensor([ [0.5,- 0.5] ,  [-0.5, 0.5],\
                                          [1,-1] ,  [0.5,  0.5],]).view(1,2,2,2))


    inputs = torch.Tensor([ [1.1,2.1],[15,.01],[1,0],[1.,1.0]]  ).view(1,2,2,2)

    # Check the result
    result = conv(inputs)
    expected_result = torch.nn.functional.conv2d(inputs,  torch.Tensor([ [sign(0.5),sign(-0.5)] ,  [sign(-0.5), sign(0.5)],\
                                          [sign(1),sign(-1)] ,  [sign(0.5),  sign(0.5)],]).view(1,2,2,2) ,None)

    assert torch.all(torch.eq(result, expected_result))


    # Redo the test with a bias
    conv2 = BinConv2d(2,1, [2,2],stride=1, bias=True)
    conv2.weight.data.copy_(torch.Tensor([ [0.5,- 0.5] ,  [-0.5, 0.5],\
                                          [1,-1] ,  [0.5,  0.5],]).view(1,2,2,2))
                                          

    conv2.bias.data.copy_(torch.Tensor([33.3]))
    result2 = conv2(inputs)
    expected_result2 = torch.nn.functional.conv2d(inputs,  torch.Tensor([ [sign(0.5),sign(-0.5)] ,  [sign(-0.5), sign(0.5)],\
                                          [sign(1),sign(-1)] ,  [sign(0.5),  sign(0.5)],]).view(1,2,2,2) ,torch.Tensor([33.3]))

    assert torch.all(torch.eq(result2, expected_result2))



def test_lin_layer_backward():
    """
    Test the backprob into layer's weight.
    """
    #Setup input and layer
    inputs = torch.autograd.Variable(torch.Tensor([2,-0.5,1]).view(1,-1), requires_grad=True)
    weight = torch.autograd.Variable(torch.Tensor([0.5,-0.5,0.5]).view(1,3), requires_grad=True)

    lin = LinearBin(3,1, bias=False)
    lin.weight.data.copy_(weight)
    
    loss = lin(inputs)
    loss.backward()

    # Check if weight grad is equal to the manual one.
    assert torch.all(torch.eq(lin.weight.grad, inputs.view(lin.weight.grad.shape)))

    #Redo test with more complicated graph.
    weight2 = torch.autograd.Variable(torch.Tensor([0.5,-0.5,0.5]).view(1,3), requires_grad=True)
    lin2 = LinearBin(3,1, bias=False)
    lin2.weight.data.copy_(weight2)

    loss2 = torch.pow(lin2(inputs),2)
    loss2.backward()  

    assert torch.all(torch.eq(lin2.weight.grad,\
                              2*inputs*torch.mm(inputs, BinaryConnectDeterministic.apply(weight.view(3,1)))))


def test_lin_layer_train():
    """
    Test the eval and train mode (dense layer).
    (We in eval mode must be quantized)
    """

    #Setup inputs and layers
    weight = torch.autograd.Variable(torch.Tensor([0.5,-0.5,0.5]).view(1,3), requires_grad=True)
    inputs = torch.autograd.Variable(torch.Tensor([2,-0.5,1]).view(1,-1), requires_grad=True)
    lin = LinearBin(3,1, bias=False)
    lin.weight.data.copy_(weight)


    # Check the forward pass
    assert torch.all(torch.eq(lin(inputs), torch.mm(inputs,   BinaryConnectDeterministic.apply(torch.transpose(weight,1,0)))))
    
    lin.train(True)
    # Check the forward pass
    assert torch.all(torch.eq(lin(inputs), torch.mm(inputs,   BinaryConnectDeterministic.apply(torch.transpose(weight,1,0)))))
    # Look at layer's weight on training mode.
    assert torch.all(torch.eq(weight, lin.weight))
    
    lin.train(False)
    # Check the forward pass
    assert torch.all(torch.eq(lin(inputs), torch.mm(inputs,   BinaryConnectDeterministic.apply(torch.transpose(weight,1,0)))))
    # Look at layer's weight on eval mode.
    assert torch.all(torch.eq(BinaryConnectDeterministic.apply( weight), lin.weight))
    
    #Redo test on eval mode.
    lin.train(True)
    assert torch.all(torch.eq(lin(inputs), torch.mm(inputs,   BinaryConnectDeterministic.apply(torch.transpose(weight,1,0)))))
    assert torch.all(torch.eq(weight, lin.weight))
    

def test_conv_layer_train():
    """
    Test the eval and train mode (conv layer).
    (We in eval mode must be quantized)
    """

    #Setup layer and input
    weight = torch.Tensor([ [0.5,- 0.5] ,  [-0.5, 0.5],\
                                          [1,-1] ,  [0.5,  0.5],]).view(1,2,2,2)
    inputs = torch.Tensor([ [1.1,2.1],[15,.01],[1,0],[1.,1.0]]  ).view(1,2,2,2)

    expected_result = torch.nn.functional.conv2d(inputs,  torch.Tensor([ [sign(0.5),sign(-0.5)] ,  [sign(-0.5), sign(0.5)],\
                                          [sign(1),sign(-1)] ,  [sign(0.5),  sign(0.5)],]).view(1,2,2,2) ,None)


    conv = BinConv2d(2,1, [2,2],stride=1, bias=False)
    conv.weight.data.copy_(weight)

    # Check the forward pass
    assert torch.all(torch.eq(expected_result, conv(inputs)))

    
    conv.train(False)
    # Check the forward pass
    assert torch.all(torch.eq(expected_result, conv(inputs)))

    # Look at layer's weight on eval mode.
    assert torch.all(torch.eq(conv.weight, BinaryConnectDeterministic.apply(weight)))

    conv.train(True)
    # Check the forward pass
    assert torch.all(torch.eq(expected_result, conv(inputs)))

    # Look at layer's weight on training mode.
    assert torch.all(torch.eq(conv.weight, weight))
