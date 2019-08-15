from QuantTorch.layers.log_lin_layers import LinearQuant, QuantConv2d
from QuantTorch.functions.log_lin_connect import LogQuant, LinQuant
import torch
import pytest
from random import randint

def equals(a, b, epsilon=1e-12):
    return torch.all(torch.lt( torch.abs(a-b), epsilon ))

@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_lin_forward_lin(fsr, bit_width, weight):

    layer = LinearQuant(weight.shape[-1], weight.shape[0], bias=False, dtype="lin", fsr=fsr, bit_width=bit_width)
    layer.weight.data.copy_(weight)

    quantOp = LinQuant(fsr, bit_width, with_sign=True)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1])

    expected = torch.mm(random_inputs, quantOp.apply(weight).transpose(1,0))
    ouputs = layer(random_inputs)

    assert equals(
        ouputs,
        expected
    )

@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_lin_forward_log(fsr, bit_width, weight):

    layer = LinearQuant(weight.shape[-1], weight.shape[0], bias=False, dtype="log", fsr=fsr, bit_width=bit_width)
    layer.weight.data.copy_(weight)

    quantOp = LogQuant(fsr, bit_width, with_sign=True)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1])

    expected = torch.mm(random_inputs, quantOp.apply(weight).transpose(1,0))
    ouputs = layer(random_inputs)

    assert equals(
        ouputs,
        expected
    )



@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("weight", [
    torch.FloatTensor(8).uniform_(-10, 10).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(-7, 8).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(50, -10).view(1,2,2,2),
])
def test_conv_forward_lin(fsr, bit_width, weight):
    
    
    layer = QuantConv2d(2,1,[2,2], stride=1, bias=False, dtype="lin", fsr=fsr, bit_width=bit_width)
    layer.weight.data.copy_(weight)


    quantOp = LinQuant(fsr, bit_width, with_sign=True)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*8).uniform_(-10,10).view(nb_sample, 2,2,2)

    expected = torch.nn.functional.conv2d(random_inputs,  quantOp.apply(weight) ,None)
    ouputs = layer(random_inputs)

    assert equals(
        ouputs,
        expected
    )

@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("weight", [
    torch.FloatTensor(8).uniform_(-10, 10).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(-7, 8).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(50, -10).view(1,2,2,2),
])
def test_conv_forward_log(fsr, bit_width, weight):
    
    
    layer = QuantConv2d(2,1,[2,2], stride=1, bias=False, dtype="log", fsr=fsr, bit_width=bit_width)
    layer.weight.data.copy_(weight)


    quantOp = LogQuant(fsr, bit_width, with_sign=True)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*8).uniform_(-10,10).view(nb_sample, 2,2,2)

    expected = torch.nn.functional.conv2d(random_inputs,  quantOp.apply(weight) ,None)
    ouputs = layer(random_inputs)

    assert equals(
        ouputs,
        expected
    )

@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_lin_train_lin(fsr, bit_width, weight):

    layer = LinearQuant(weight.shape[-1], weight.shape[0], bias=False, dtype="lin", fsr=fsr, bit_width=bit_width)
    layer.weight.data.copy_(weight)

    quantOp = LinQuant(fsr, bit_width, with_sign=True)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1])

    expected = torch.mm(random_inputs, quantOp.apply(weight).transpose(1,0))
    ouputs = layer(random_inputs)

    assert equals(
        ouputs,
        expected
    )

    assert equals(
        layer.weight,
        weight
    )

    layer.train(False)

    assert equals(
        layer.weight,
        quantOp.apply(weight)
    )
    
    ouputs = layer(random_inputs)
    assert equals(
        ouputs,
        expected
    )


    layer.train(True)

    assert equals(
        layer.weight,
        weight
    )
    
    ouputs = layer(random_inputs)
    assert equals(
        ouputs,
        expected
    )



@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("weight", [
    torch.FloatTensor(8).uniform_(-10, 10).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(-7, 8).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(50, -10).view(1,2,2,2),
])
def test_conv_train_lin(fsr, bit_width, weight):
    
    
    layer = QuantConv2d(2,1,[2,2], stride=1, bias=False, dtype="log", fsr=fsr, bit_width=bit_width)
    layer.weight.data.copy_(weight)


    quantOp = LogQuant(fsr, bit_width, with_sign=True)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*8).uniform_(-10,10).view(nb_sample, 2,2,2)

    expected = torch.nn.functional.conv2d(random_inputs,  quantOp.apply(weight) ,None)
    ouputs = layer(random_inputs)

    assert equals(
        ouputs,
        expected
    )
    assert equals(
        layer.weight,
        weight
    )

    layer.train(False)

    ouputs = layer(random_inputs)

    assert equals(
        ouputs,
        expected
    )
    assert equals(
        layer.weight,
        quantOp.apply(weight)
    )

    layer.train(True)

    ouputs = layer(random_inputs)

    assert equals(
        ouputs,
        expected
    )
    assert equals(
        layer.weight,
        weight
    )

