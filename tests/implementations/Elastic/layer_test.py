from QuantTorch.functions.elastic_quant_connect import _proj_val, lin_proj, exp_proj,\
    lin_deriv_l2, exp_deriv_l2, lin_deriv_l1, exp_deriv_l1, QuantWeightLin, QuantWeightExp
from QuantTorch.layers.elastic_layers import LinearQuantLin, LinearQuantLog, QuantConv2dLin, QuantConv2dLog 
from random import randint

import pytest
import torch

def equals(a, b, epsilon=1e-5):
    return torch.all(torch.lt( torch.abs(a-b), epsilon ))

@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_forward_lin_dense(weight):
    layer_l1  = LinearQuantLin(weight.shape[-1], weight.shape[0],bias=False, alpha=0, beta=1)
    layer_l2  = LinearQuantLin(weight.shape[-1], weight.shape[0],bias=False, alpha=1, beta=0)
    layer_mix = LinearQuantLin(weight.shape[-1], weight.shape[0],bias=False, alpha=1, beta=1)

    layer_l1.weight.data.copy_(weight)
    layer_l2.weight.data.copy_(weight)
    layer_mix.weight.data.copy_(weight)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1])

    expected = torch.mm(random_inputs, weight.transpose(1,0))
    ouputs_l1  = layer_l1(random_inputs)
    ouputs_l2  = layer_l2(random_inputs)
    ouputs_mix = layer_mix(random_inputs)


    assert equals(expected, ouputs_l1,  1e-5)
    assert equals(expected, ouputs_l2,  1e-5)
    assert equals(expected, ouputs_mix, 1e-5)


@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_forward_log_dense(weight):
    layer_l1  = LinearQuantLog(weight.shape[-1], weight.shape[0],bias=False, alpha=0, beta=1)
    layer_l2  = LinearQuantLog(weight.shape[-1], weight.shape[0],bias=False, alpha=1, beta=0)
    layer_mix = LinearQuantLog(weight.shape[-1], weight.shape[0],bias=False, alpha=1, beta=1)

    layer_l1.weight.data.copy_(weight)
    layer_l2.weight.data.copy_(weight)
    layer_mix.weight.data.copy_(weight)


    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1])

    expected = torch.mm(random_inputs, weight.transpose(1,0))
    ouputs_l1  = layer_l1(random_inputs)
    ouputs_l2  = layer_l2(random_inputs)
    ouputs_mix = layer_mix(random_inputs)


    assert equals(expected, ouputs_l1,  1e-5)
    assert equals(expected, ouputs_l2,  1e-5)
    assert equals(expected, ouputs_mix, 1e-5)


@pytest.mark.parametrize("weight", [
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(0, -1).view(1,2,2,2),
])
def test_forward_lin_conv(weight):
    layer_l1  = QuantConv2dLin(2,1,[2,2], stride=1, bias=False, alpha=0, beta=1)
    layer_l2  = QuantConv2dLin(2,1,[2,2], stride=1, bias=False, alpha=1, beta=0)
    layer_mix = QuantConv2dLin(2,1,[2,2], stride=1, bias=False, alpha=1, beta=1)

    layer_l1.weight.data.copy_(weight)
    layer_l2.weight.data.copy_(weight)
    layer_mix.weight.data.copy_(weight)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*8).uniform_(-10,10).view(nb_sample, 2,2,2)

    expected = torch.nn.functional.conv2d(random_inputs,  weight, None, padding=1)
    ouputs_l1  = layer_l1(random_inputs)
    ouputs_l2  = layer_l2(random_inputs)
    ouputs_mix = layer_mix(random_inputs)

    assert equals(
        expected,
        ouputs_l1
    )
    assert equals(
        expected,
        ouputs_l2
    )
    assert equals(
        expected,
        ouputs_mix
    )

@pytest.mark.parametrize("weight", [
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(0, -1).view(1,2,2,2),
])
def test_forward_log_conv(weight):
    layer_l1  = QuantConv2dLog(2,1,[2,2], stride=1, bias=False, alpha=0, beta=1)
    layer_l2  = QuantConv2dLog(2,1,[2,2], stride=1, bias=False, alpha=1, beta=0)
    layer_mix = QuantConv2dLog(2,1,[2,2], stride=1, bias=False, alpha=1, beta=1)

    layer_l1.weight.data.copy_(weight)
    layer_l2.weight.data.copy_(weight)
    layer_mix.weight.data.copy_(weight)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*8).uniform_(-10,10).view(nb_sample, 2,2,2)

    expected = torch.nn.functional.conv2d(random_inputs,  weight, None, padding=1)
    ouputs_l1  = layer_l1(random_inputs)
    ouputs_l2  = layer_l2(random_inputs)
    ouputs_mix = layer_mix(random_inputs)

    assert equals(
        expected,
        ouputs_l1
    )
    assert equals(
        expected,
        ouputs_l2
    )
    assert equals(
        expected,
        ouputs_mix
    )

@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-1, 1).view(3,2),
    torch.FloatTensor(5).uniform_(-1, 1).view(1,5),
    torch.FloatTensor(10).uniform_(-1, 1).view(2,5),
])
def test_backprob_lin_dense(weight):
    layer_l1  = LinearQuantLin(weight.shape[-1], weight.shape[0],bias=False, alpha=0, beta=1, size=5, bottom=-1, top=1)
    layer_l2  = LinearQuantLin(weight.shape[-1], weight.shape[0],bias=False, alpha=1, beta=0, size=5, bottom=-1, top=1)
    layer_mix = LinearQuantLin(weight.shape[-1], weight.shape[0],bias=False, alpha=1, beta=1, size=5, bottom=-1, top=1)

    layer_l1.weight.data.copy_(weight)
    layer_l2.weight.data.copy_(weight)
    layer_mix.weight.data.copy_(weight)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1])

    loss_l1  = torch.sum(torch.pow(layer_l1(random_inputs),2))
    loss_l2  = torch.sum(torch.pow(layer_l2(random_inputs),2))
    loss_mix = torch.sum(torch.pow(layer_mix(random_inputs),2))

    loss_l1.backward()
    loss_l2.backward()
    loss_mix.backward()

    assert( equals(
            layer_l1.weight.grad,
            torch.mm(2*torch.mm(random_inputs, weight.transpose(1,0)).transpose(1,0),random_inputs)
             - lin_deriv_l1(weight, 1, top=1, bottom=-1, size=5)
    ))
    assert( equals(
            layer_l2.weight.grad,
            torch.mm(2*torch.mm(random_inputs, weight.transpose(1,0)).transpose(1,0),random_inputs)
             - lin_deriv_l2(weight, 1, top=1, bottom=-1, size=5)
    ))

    assert( equals(
            layer_mix.weight.grad,
            torch.mm(2*torch.mm(random_inputs, weight.transpose(1,0)).transpose(1,0),random_inputs)
             - lin_deriv_l1(weight, 1, top=1, bottom=-1, size=5)
             - lin_deriv_l2(weight, 1, top=1, bottom=-1, size=5)
    ))

@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-1, 1).view(3,2),
    torch.FloatTensor(5).uniform_(-1, 1).view(1,5),
    torch.FloatTensor(10).uniform_(-1, 1).view(2,5),
])
def test_backprob_log_dense(weight):
    layer_l1  = LinearQuantLog(weight.shape[-1], weight.shape[0],bias=False, alpha=0, beta=1, gamma=2, init=0.25, size=5)
    layer_l2  = LinearQuantLog(weight.shape[-1], weight.shape[0],bias=False, alpha=1, beta=0, gamma=2, init=0.25, size=5)
    layer_mix = LinearQuantLog(weight.shape[-1], weight.shape[0],bias=False, alpha=1, beta=1, gamma=2, init=0.25, size=5)

    layer_l1.weight.data.copy_(weight)
    layer_l2.weight.data.copy_(weight)
    layer_mix.weight.data.copy_(weight)

    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*weight.shape[-1]).uniform_(-10,10).view(nb_sample, weight.shape[-1])

    loss_l1  = torch.sum(torch.pow(layer_l1(random_inputs),2))
    loss_l2  = torch.sum(torch.pow(layer_l2(random_inputs),2))
    loss_mix = torch.sum(torch.pow(layer_mix(random_inputs),2))

    loss_l1.backward()
    loss_l2.backward()
    loss_mix.backward()

    assert( equals(
            layer_l1.weight.grad,
            torch.mm(2*torch.mm(random_inputs, weight.transpose(1,0)).transpose(1,0),random_inputs)
             - exp_deriv_l1(weight, 1,  gamma=2, init=0.25, size=5)
    ))
    assert( equals(
            layer_l2.weight.grad,
            torch.mm(2*torch.mm(random_inputs, weight.transpose(1,0)).transpose(1,0),random_inputs)
             - exp_deriv_l2(weight, 1,  gamma=2, init=0.25, size=5)
    ))

    assert( equals(
            layer_mix.weight.grad,
            torch.mm(2*torch.mm(random_inputs, weight.transpose(1,0)).transpose(1,0),random_inputs)
             - exp_deriv_l1(weight, 1, gamma=2, init=0.25, size=5)
             - exp_deriv_l2(weight, 1, gamma=2, init=0.25, size=5)
    ))


@pytest.mark.parametrize("weight", [
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(0, -1).view(1,2,2,2),
])
def test_backprob_lin_conv(weight):
    layer_l1  = QuantConv2dLin(2,1,[2,2], stride=1, bias=False, alpha=0, beta=1)
    layer_l2  = QuantConv2dLin(2,1,[2,2], stride=1, bias=False, alpha=1, beta=0)
    layer_mix = QuantConv2dLin(2,1,[2,2], stride=1, bias=False, alpha=1, beta=1)


    layer_l1.weight.data.copy_(weight)
    layer_l2.weight.data.copy_(weight)
    layer_mix.weight.data.copy_(weight)


    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*8).uniform_(-10,10).view(nb_sample, 2,2,2)

    loss_l1  = torch.sum(torch.pow(layer_l1(random_inputs),2))
    loss_l2  = torch.sum(torch.pow(layer_l2(random_inputs),2))
    loss_mix = torch.sum(torch.pow(layer_mix(random_inputs),2))

    loss_l1.backward()
    loss_l2.backward()
    loss_mix.backward()

    assert equals(
        layer_l1.weight.grad,
        torch.nn.grad.conv2d_weight(random_inputs, weight.shape, 2*layer_l1(random_inputs), padding=1)
         - lin_deriv_l1(weight, 1),
         1e-3
    )
    assert equals(
        layer_l2.weight.grad,
        torch.nn.grad.conv2d_weight(random_inputs, weight.shape, 2*layer_l2(random_inputs), padding=1)
         - lin_deriv_l2(weight, 1),
         1e-3
    )    
    assert equals(
        layer_mix.weight.grad,
        torch.nn.grad.conv2d_weight(random_inputs, weight.shape, 2*layer_mix(random_inputs), padding=1)
         - lin_deriv_l1(weight, 1)
         - lin_deriv_l2(weight, 1),
         1e-3
    )


@pytest.mark.parametrize("weight", [
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(-1, 1).view(1,2,2,2),
    torch.FloatTensor(8).uniform_(0, -1).view(1,2,2,2),
])
def test_backprob_log_conv(weight):
    layer_l1  = QuantConv2dLog(2,1,[2,2], stride=1, bias=False, alpha=0, beta=1, gamma=2, init=0.25, size=5)
    layer_l2  = QuantConv2dLog(2,1,[2,2], stride=1, bias=False, alpha=1, beta=0, gamma=2, init=0.25, size=5)
    layer_mix = QuantConv2dLog(2,1,[2,2], stride=1, bias=False, alpha=1, beta=1, gamma=2, init=0.25, size=5)


    layer_l1.weight.data.copy_(weight)
    layer_l2.weight.data.copy_(weight)
    layer_mix.weight.data.copy_(weight)


    nb_sample = randint(1,10)
    random_inputs = torch.FloatTensor(nb_sample*8).uniform_(-10,10).view(nb_sample, 2,2,2)

    loss_l1  = torch.sum(torch.pow(layer_l1(random_inputs),2))
    loss_l2  = torch.sum(torch.pow(layer_l2(random_inputs),2))
    loss_mix = torch.sum(torch.pow(layer_mix(random_inputs),2))

    loss_l1.backward()
    loss_l2.backward()
    loss_mix.backward()
    print(layer_l1.weight.grad)
    print(torch.nn.grad.conv2d_weight(random_inputs, weight.shape, 2*layer_l1(random_inputs), padding=1)
         - exp_deriv_l1(weight, 1, gamma=2, init=0.25, size=5))
    assert equals(
        layer_l1.weight.grad,
        torch.nn.grad.conv2d_weight(random_inputs, weight.shape, 2*layer_l1(random_inputs), padding=1)
         - exp_deriv_l1(weight, 1, gamma=2, init=0.25, size=5),
         1e-3
    )
    assert equals(
        layer_l2.weight.grad,
        torch.nn.grad.conv2d_weight(random_inputs, weight.shape, 2*layer_l2(random_inputs), padding=1)
         - exp_deriv_l2(weight, 1, gamma=2, init=0.25, size=5),
         1e-3
    )    
    assert equals(
        layer_mix.weight.grad,
        torch.nn.grad.conv2d_weight(random_inputs, weight.shape, 2*layer_mix(random_inputs), padding=1)
         - exp_deriv_l1(weight, 1, gamma=2, init=0.25, size=5)
         - exp_deriv_l2(weight, 1, gamma=2, init=0.25, size=5),
         1e-3
    )
