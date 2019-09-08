from QuantTorch.functions.terner_connect import  TernaryConnectDeterministic, TernaryConnectStochastic
import torch 
import pytest 

def equals(a, b, epsilon=1e-12):
    return torch.all(torch.lt( torch.abs(a-b), epsilon ))



def test_terner_connect_det_forward():
    x_1 = torch.Tensor([0.75,0.5,0.25,0.0,-1,-0.2])
    x_2 = torch.Tensor([1,0,0.51,0.1,0,-1,-0.2,.7]).view(2,4)

    y_1_expected = torch.Tensor([1,1,0,0,-1,0])
    y_2_expected = torch.Tensor([1,0,1,0.,0,-1,0,1]).view(2,4)

    y_1 = TernaryConnectDeterministic.apply(x_1)
    y_2 = TernaryConnectDeterministic.apply(x_2)

    assert equals(
        y_1,
        y_1_expected
    )
    assert equals(
        y_2,
        y_2_expected
    )


def test_terner_connect_sto_forward():
    x = torch.Tensor([1,0,0.45,-1,-0.9]).view(1,-1)

    results = list()
    for i in range(1000):
        temp_result = TernaryConnectStochastic.apply(x)
        # Tensor must have only -1 , 0 , 1 values
        assert not torch.any(torch.lt(torch.abs(temp_result-1),1e-8)*torch.lt(torch.abs(temp_result),1e-8))
        results.append(temp_result) 

    result = torch.cat(results,0 )
    result = torch.sum(result, 0)/1000
    
    assert equals(
        result,
        torch.Tensor([1,0,0.45,-1,-0.9]).view(1,-1),
        5e-2)


@pytest.mark.parametrize("inputs, weight", [
    [torch.FloatTensor(6).uniform_(-10, 10).view(3,2),torch.FloatTensor(6).uniform_(-1, 1).view(2,3)],
    [torch.FloatTensor(5).uniform_(-7, 8).view(1,5),torch.FloatTensor(5).uniform_(-1,1).view(5,1)],
    [torch.FloatTensor(10).uniform_(50, -10).view(2,5),torch.FloatTensor(5).uniform_(-1,1).view(5,1)]
])
def test_terner_connect_det_backward(inputs, weight):
    
    #setup all vars
    inputs_var_1 = torch.autograd.Variable(inputs, requires_grad=True)
    weight_var_1 = torch.autograd.Variable(weight, requires_grad=True)

    inputs_var_2 = torch.autograd.Variable(inputs, requires_grad=True)
    weight_var_2 = torch.autograd.Variable(weight, requires_grad=True)


    loss_1 = torch.sum(torch.mm(inputs,TernaryConnectDeterministic.apply(weight_var_1) ))
    loss_1.backward()

    assert equals(
        weight_var_1.grad,
        torch.transpose(torch.sum(inputs_var_1,0, keepdim=True),1,0).repeat(1, weight.shape[-1])
    )

    
    loss_2_temp = torch.mm(inputs_var_2,TernaryConnectDeterministic.apply(weight_var_2) )
    loss_2 = torch.sum(torch.pow(loss_2_temp, 2))
    loss_2.backward()

    assert equals(
        weight_var_2.grad,
        torch.mm(inputs_var_2.transpose(1,0),2*loss_2_temp )
    )

def test_terner_connect_det_backward_bis():
    x = torch.autograd.Variable(torch.Tensor([2,1.0,0.0,-1,-3]), requires_grad=True)
    
    loss = torch.sum(TernaryConnectDeterministic.apply(x))
    loss.backward()

    assert equals(
        x.grad[0],
        0)

    assert equals(
        x.grad[4],
        0)


@pytest.mark.parametrize("inputs, weight", [
    [torch.FloatTensor(6).uniform_(-10, 10).view(3,2),torch.FloatTensor(6).uniform_(-1, 1).view(2,3)],
    [torch.FloatTensor(5).uniform_(-7, 8).view(1,5),torch.FloatTensor(5).uniform_(-1,1).view(5,1)],
    [torch.FloatTensor(10).uniform_(50, -10).view(2,5),torch.FloatTensor(5).uniform_(-1,1).view(5,1)]
])
def test_terner_connect_sto_backward(inputs, weight):
    
    #setup all vars
    inputs_var_1 = torch.autograd.Variable(inputs, requires_grad=True)
    weight_var_1 = torch.autograd.Variable(weight, requires_grad=True)

    inputs_var_2 = torch.autograd.Variable(inputs, requires_grad=True)
    weight_var_2 = torch.autograd.Variable(weight, requires_grad=True)


    loss_1 = torch.sum(torch.mm(inputs,TernaryConnectStochastic.apply(weight_var_1) ))
    loss_1.backward()

    assert equals(
        weight_var_1.grad,
        torch.transpose(torch.sum(inputs_var_1,0, keepdim=True),1,0).repeat(1, weight.shape[-1])
    )

    loss_2_temp = torch.mm(inputs_var_2,TernaryConnectStochastic.apply(weight_var_2) )
    loss_2 = torch.sum(torch.pow(loss_2_temp, 2))
    loss_2.backward()

    assert equals(
        weight_var_2.grad,
        torch.mm(inputs_var_2.transpose(1,0),2*loss_2_temp )
    )
