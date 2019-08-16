from QuantTorch.functions.terner_connect import  TernaryConnectDeterministic, TernaryConnectStochastic
from QuantTorch.layers.terner_layers    import LinearTer, TerConv2d
import torch 
import pytest 
from random import randint

def equals(a, b, epsilon=1e-12):
    return torch.all(torch.lt( torch.abs(a-b), epsilon ))



@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_lin_forward(weight):
    #setup layer
    layer_det = LinearTer(weight.shape[-1], weight.shape[0], deterministic=True)
    layer_det.weight.data.copy_(weight)

    quantOp = TernaryConnectDeterministic

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1])

    expected = torch.mm(random_inputs, quantOp.apply(weight).transpose(1,0))
    ouputs = layer_det(random_inputs)

    assert equals(
            expected,
            ouputs
    )


    layer_sto = LinearTer(weight.shape[-1], weight.shape[0], deterministic=False)
    layer_sto.weight.data.copy_(torch.ones_like(weight)*0.5)


    assert not torch.all(torch.eq(
        layer_sto(random_inputs),
        layer_sto(random_inputs)
    ))


@pytest.mark.parametrize("weight", [
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(0, -1).view(1,2,2,2),
])
def test_conv_forward(weight):
    
    layer_det = TerConv2d(2,1,[2,2], stride=1, bias=False, deterministic=True)
    layer_det.weight.data.copy_(weight)

    quantOp = TernaryConnectDeterministic


    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*8).uniform_(-10,10).view(nb_sample, 2,2,2)

    expected = torch.nn.functional.conv2d(random_inputs,  quantOp.apply(weight) ,None)
    ouputs = layer_det(random_inputs)

    assert equals(
        ouputs,
        expected
    )


    layer_sto = TerConv2d(2,1,[2,2], stride=1, bias=False, deterministic=False)
    layer_sto.weight.data.copy_(torch.ones_like(weight)*0.5)


    assert not torch.all(torch.eq(
        layer_sto(random_inputs),
        layer_sto(random_inputs)
    ))


@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_lin_train(weight):
    #setup layer
    layer_det = LinearTer(weight.shape[-1], weight.shape[0], deterministic=True)
    layer_det.weight.data.copy_(weight)

    quantOp = TernaryConnectDeterministic

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1])

    expected = torch.mm(random_inputs, quantOp.apply(weight).transpose(1,0))
    ouputs = layer_det(random_inputs)

    assert equals(
            expected,
            ouputs
    )

    assert equals(
        weight,
        layer_det.weight
    )

    layer_det.train(False)

    assert equals(
            expected,
            ouputs
    )
    assert equals(
        quantOp.apply(weight),
        layer_det.weight
    )

    layer_det.train(True)

    assert equals(
            expected,
            ouputs
    )
    assert equals(
        weight,
        layer_det.weight
    )




@pytest.mark.parametrize("weight", [
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(0, -1).view(1,2,2,2),
])
def test_conv_train(weight):
    
    layer_det = TerConv2d(2,1,[2,2], stride=1, bias=False, deterministic=True)
    layer_det.weight.data.copy_(weight)

    quantOp = TernaryConnectDeterministic


    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*8).uniform_(-10,10).view(nb_sample, 2,2,2)

    expected = torch.nn.functional.conv2d(random_inputs,  quantOp.apply(weight) ,None)
    ouputs = layer_det(random_inputs)

    assert equals(
        ouputs,
        expected
    )

    assert equals(
        weight,
        layer_det.weight
    )

    layer_det.train(False)

    assert equals(
        ouputs,
        expected
    )

    assert equals(
        quantOp.apply(weight),
        layer_det.weight
    )

    layer_det.train(True)

    assert equals(
        ouputs,
        expected
    )

    assert equals(
        weight,
        layer_det.weight
    )



@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-1, 1).view(3,2),
    torch.FloatTensor(5).uniform_(-1, 1).view(1,5),
    torch.FloatTensor(10).uniform_(-1, 1).view(2,5),
])
def test_lin_backward_1(weight):
    #setup layer
    
    layer = LinearTer(weight.shape[-1], weight.shape[0], deterministic=True)
    layer.weight.data.copy_(weight)

    quantOp = TernaryConnectDeterministic

    nb_sample = randint(1,10)
    random_inputs = torch.autograd.Variable(torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1]), requires_grad=True)

    loss = torch.sum(layer(random_inputs))

    loss.backward()

    assert equals(
        layer.weight.grad,
        torch.sum(random_inputs,0, keepdim=True).repeat(layer.weight.shape[0],1),
        1e-5
    )

@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-1, 1).view(3,2),
    torch.FloatTensor(5).uniform_(-1, 1).view(1,5),
    torch.FloatTensor(10).uniform_(-1, 1).view(2,5),
])
def test_lin_backward_2(weight):
    #setup layer
    
    layer = LinearTer(weight.shape[-1], weight.shape[0], deterministic=True)
    layer.weight.data.copy_(weight)

    quantOp = TernaryConnectDeterministic

    nb_sample = randint(1,10)
    random_inputs = torch.autograd.Variable(torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1]), requires_grad=True)

    layer_output = layer(random_inputs)
    loss = torch.sum(torch.pow(layer_output, 2))
    loss.backward()


    assert equals(
        layer.weight.grad,
        torch.mm(2*layer_output.transpose(1,0), random_inputs)
    )

    assert equals(
        random_inputs.grad,
        torch.mm(2*layer_output, quantOp.apply(weight))
    )


def test_lin_backward_3():
    #setup layer
    weight = torch.Tensor([[1,4,0,3]])

    layer = LinearTer(weight.shape[-1], weight.shape[0], deterministic=True)
    layer.weight.data.copy_(weight)

    quantOp = TernaryConnectDeterministic

    nb_sample = randint(1,10)
    random_inputs = torch.autograd.Variable(torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1]), requires_grad=True)

    layer_output = layer(random_inputs)
    loss = torch.sum(torch.pow(layer_output, 2))
    loss.backward()

    assert equals(
        layer.weight.grad[0,1],
        0.
    )

    assert equals(
        layer.weight.grad[0,3],
        0.
    )

def test_conv_backward():
    #TODO 
    pass

