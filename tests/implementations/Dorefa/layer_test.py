from QuantTorch.layers import DorefaConv2d, LinearDorefa
from QuantTorch.functions import DorefaQuant, nnQuantWeight
import torch
import pytest
from random import randint

def equals(a, b, epsilon=1e-12):
    return torch.all(torch.lt( torch.abs(a-b), epsilon ))



def test_lin_layer_train():
    """
    Test the eval and train mode (dense layer).
    (Weight in eval mode must be quantized)
    """

    #Setup inputs and layers
    weight = torch.autograd.Variable(torch.Tensor([0.5,-0.5,0.5]).view(1,3), requires_grad=True)
    q_weight = nnQuantWeight( bit_width=3)(weight)
    inputs = torch.autograd.Variable(torch.Tensor([2,-0.5,1]).view(1,-1), requires_grad=True)
    lin = LinearDorefa(3,1, bias=False, bit_width=3)
    lin.weight.data.copy_(weight)


    # Check the forward pass
    assert torch.all(torch.eq(lin(inputs), torch.mm(inputs,q_weight.transpose(1,0) )))

    lin.train(True)
    # Check the forward pass
    assert torch.all(torch.eq(lin(inputs), torch.mm(inputs,  q_weight.transpose(1,0))))
    # Look at layer's weight on training mode.
    assert torch.all(torch.eq(weight, lin.weight))
    
    lin.train(False)
    # Check the forward pass
    assert torch.all(torch.eq(lin(inputs), torch.mm(inputs,q_weight.transpose(1,0))))
    # Look at layer's weight on eval mode.
    assert torch.all(torch.eq(q_weight, lin.weight))
    
    #Redo test on eval mode.
    lin.train(True)
    assert torch.all(torch.eq(lin(inputs), torch.mm(inputs,q_weight.transpose(1,0))))
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

    q_weight =  nnQuantWeight( bit_width=3)(weight)


    expected_result = torch.nn.functional.conv2d(inputs, q_weight ,None, stride=1, padding=0, dilation=1, groups=1)

    conv = DorefaConv2d(2,1, [2,2], bias=False, bit_width=3, stride=1, padding=0, dilation=1, groups=1)
    conv.weight.data.copy_(weight)

    # Check the forward pass
    assert torch.all(torch.eq(expected_result, conv(inputs)))

    conv.train(False)
    # Check the forward pass
    assert torch.all(torch.eq(expected_result, conv(inputs)))

    # Look at layer's weight on eval mode.
    assert torch.all(torch.eq(conv.weight, q_weight))

    conv.train(True)
    # Check the forward pass
    assert torch.all(torch.eq(expected_result, conv(inputs)))

    # Look at layer's weight on training mode.
    assert torch.all(torch.eq(conv.weight, weight))


@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_lin_backprob(weight):
    #Generate random linear input
    rand_sample = randint(1,10)
    random_input = torch.FloatTensor(rand_sample*weight.shape[-1]).uniform_(-10,10).view(rand_sample, weight.shape[-1])
    
    lin_shape = list(weight.shape)
    lin_shape.reverse()

    lin_1b  = LinearDorefa( *lin_shape, bias=False, bit_width=1)
    lin_1b.weight.data.copy_(weight)

    lin_2b  = LinearDorefa( *lin_shape, bias=False, bit_width=2)
    lin_2b.weight.data.copy_(weight)

    lin_3b  = LinearDorefa( *lin_shape, bias=False, bit_width=3)
    lin_3b.weight.data.copy_(weight)

    lin_32b = LinearDorefa( *lin_shape, bias=False, bit_width=32)
    lin_32b.weight.data.copy_(weight)



    # Quantification op previously checked 
    weightQ1b  = nnQuantWeight(1 )
    weightQ2b  = nnQuantWeight(2 )
    weightQ3b  = nnQuantWeight(3 )
    weightQ32b = nnQuantWeight(32)


    # Convert all data to var data.
    random_input_var = torch.autograd.Variable(random_input, requires_grad=True)

    weight_1b = torch.autograd.Variable(weight, requires_grad=True)
    weight_2b = torch.autograd.Variable(weight, requires_grad=True)
    weight_2b_ = torch.autograd.Variable(weight, requires_grad=True)

    weight_3b = torch.autograd.Variable(weight, requires_grad=True)
    weight_3b_ = torch.autograd.Variable(weight, requires_grad=True)

    weight_32b = torch.autograd.Variable(weight, requires_grad=True)


    # Test 1 b
    loss_1b = torch.sum(lin_1b(random_input_var))
    loss_1b.backward()

    assert equals(
        lin_1b.weight.grad,
        torch.sum(random_input_var,0).view(1,-1).repeat(weight.shape[0],1),
        1e-5
    )

    # Test 2 b
    loss_2b = torch.sum(lin_2b(random_input_var))
    loss_2b.backward()

    loss_2b_ = torch.sum(torch.mm(random_input_var, weightQ2b(weight_2b_).transpose(1,0)))
    loss_2b_.backward()

    assert equals(
        lin_2b.weight.grad,
        weight_2b_.grad,
        1e-5
    )

    # Test 3 b
    loss_3b = torch.sum(lin_3b(random_input_var))
    loss_3b.backward()

    loss_3b_ = torch.sum(torch.mm(random_input_var, weightQ3b(weight_3b_).transpose(1,0)))
    loss_3b_.backward()

    assert equals(
        lin_3b.weight.grad,
        weight_3b_.grad,
        1e-5
    )

    # Test 32 b
    loss_32b = torch.sum(lin_32b(random_input_var))
    loss_32b.backward()

    assert equals(
        lin_32b.weight.grad,
        torch.sum(random_input_var,0).view(1,-1).repeat(weight.shape[0],1),
        1e-5
    )



def test_conv_backprob():
    #TODO 
    pass

if __name__ == "__main__":
    test_lin_backprob(torch.FloatTensor(6).uniform_(-10, 10).view(3,2))