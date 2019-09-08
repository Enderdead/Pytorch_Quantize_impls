from QuantTorch.functions.elastic_quant_connect import _proj_val, lin_proj, exp_proj,\
    lin_deriv_l2, exp_deriv_l2, lin_deriv_l1, exp_deriv_l1, QuantWeightLin, QuantWeightExp

import pytest
import torch


def equals(a, b, epsilon=1e-12):
    return torch.all(torch.lt( torch.abs(a-b), epsilon ))

def test_proj_val():
    x1 = torch.Tensor([00.1,1,4,-4,-7,0.1,4])
    sets1 = torch.Tensor([1,0,-1])

    assert equals(
        _proj_val(x1, sets1),
        torch.Tensor([ 0.,  1.,  1., -1., -1.,  0.,  1.])
    )

    x2 = torch.Tensor([0.5,1,44,-1,-0.7,-10,45])
    sets2 = torch.Tensor([0,-1])

    assert equals(
        _proj_val(x2, sets2),
        torch.Tensor([ 0.,  0.,  0., -1., -1.,  -1.,  0.])
    )
    x3 = torch.Tensor([-1,-2,-4,-4,4,6])
    sets3 = torch.Tensor([-50])

    assert equals(
        _proj_val(x3, sets3),
        torch.Tensor([-50,-50,-50,-50,-50,-50])
    )


@pytest.mark.parametrize("alpha", [1,2,3,0.5,0.03])
def test_lin_deriv_l2(alpha):

    x1 = torch.Tensor([0.0,0.5,1.0])
    x2 = torch.Tensor([0.1,0.6,0.9])
    x3 = torch.Tensor([0.2,0.5,0.75])


    func = lambda x : lin_deriv_l2(x, alpha, top=1, bottom=0, size=3) # 0 0.5 1
    
    assert equals(
        func(x1),
        torch.zeros_like(x1)
    )

    assert equals(
        func(x2),
        torch.Tensor([0.1*alpha, 0.1*alpha, -0.1*alpha]),
        1e-5
    )
    assert equals(
        func(x3),
        torch.Tensor([0.2*alpha,  0., 0.25*alpha]),
        1e-5
    )


@pytest.mark.parametrize("alpha", [1,2,3,0.5,0.03])
def test_exp_deriv_l2(alpha):
    x1 = torch.Tensor([-1.0,-0.5,0.5,1.0])
    x2 = torch.Tensor([-0.8,-0.6,0.1,0.8])
    x3 = torch.Tensor([0.75,0.0,-1])


    func = lambda x : exp_deriv_l2(x, alpha, gamma=2, init=.5, size=2) # -1 -0.5 0.5 1
    
    assert equals(
        func(x1),
        torch.zeros_like(x1)
    )

    assert equals(
        func(x2),
        torch.Tensor([0.2*alpha, -0.1*alpha,-0.4*alpha,-0.2*alpha]),
        1e-5
    )   

    assert equals(
        func(x3),
        torch.Tensor([0.25*alpha,  0.5*alpha, 0.0]),
        1e-5
    )


@pytest.mark.parametrize("alpha", [1,2,3,0.5,0.03])
def test_lin_deriv_l1(alpha):
    x1 = torch.Tensor([0.0,0.5,1.0])
    x2 = torch.Tensor([0.1,0.6,0.9])
    x3 = torch.Tensor([0.2,0.5,0.75])

    func = lambda x : lin_deriv_l1(x, alpha, top=1, bottom=0, size=3)
    
    assert equals(
        func(x1),
        torch.zeros_like(x1)
    )

    assert equals(
        func(x2),
        torch.Tensor([1*alpha, 1*alpha, -1*alpha]),
        1e-5
    )
    assert equals(
        func(x3),
        torch.Tensor([+1*alpha,  0., +1*alpha]),
        1e-5
    )


@pytest.mark.parametrize("alpha", [1,2,3,0.5,0.03])
def test_exp_deriv_l1(alpha):
    x1 = torch.Tensor([-1.0,-0.5,0.5,1.0])
    x2 = torch.Tensor([-0.8,-0.6,0.1,0.8])
    x3 = torch.Tensor([0.75,0.0,-1, -0.75])

    func = lambda x : exp_deriv_l1(x, alpha, gamma=2, init=.5, size=2)
    assert equals(
        func(x1),
        torch.zeros_like(x1)
    )
    assert equals(
        func(x2),
        torch.Tensor([1*alpha, -1*alpha,-1*alpha,-1*alpha]),
        1e-5
    ) 
    assert equals(
        func(x3),
        torch.Tensor([1*alpha,  1*alpha, 0.0, 1*alpha]),
        1e-5
    )



@pytest.mark.parametrize("top", [1,2,3,4])
@pytest.mark.parametrize("bottom", [-1,-2,-3,-4])
@pytest.mark.parametrize("size", [2,3,4,5])
def test_QuantWeightLin(top,  bottom, size):


    random_tensor_1 = torch.autograd.Variable(torch.FloatTensor(6).view(3,2).uniform_(-5,5),requires_grad=True)
    random_tensor_2 = torch.autograd.Variable(torch.FloatTensor(6).view(3,2).uniform_(-5,5),requires_grad=True)
    random_tensor_3 = torch.autograd.Variable(torch.FloatTensor(6).view(3,2).uniform_(-5,5),requires_grad=True)

    op = QuantWeightLin(top, bottom, size)


    # Test L2
    loss1 = torch.sum(op.apply(random_tensor_1,torch.Tensor([1]),torch.Tensor([0])))
    loss1.backward()

    assert equals(
        random_tensor_1.grad,
        torch.ones_like(random_tensor_1)-lin_deriv_l2(random_tensor_1, 1, top, bottom, size)
    )


    # Test L1
    loss2 = torch.sum(op.apply(random_tensor_2,torch.Tensor([0]),torch.Tensor([1])))
    loss2.backward()

    assert equals(
        random_tensor_2.grad,
        torch.ones_like(random_tensor_2)-lin_deriv_l1(random_tensor_2, 1, top, bottom, size)
    )


@pytest.mark.parametrize("init", [0.2,0.25,0.5,1])
@pytest.mark.parametrize("gamma", [2,1.5,4])
@pytest.mark.parametrize("size", [2,3,4,5])
def test_QuantWeightExp(init, gamma, size):

    random_tensor_1 = torch.autograd.Variable(torch.FloatTensor(6).view(3,2).uniform_(-5,5),requires_grad=True)
    random_tensor_2 = torch.autograd.Variable(torch.FloatTensor(6).view(3,2).uniform_(-5,5),requires_grad=True)
    random_tensor_3 = torch.autograd.Variable(torch.FloatTensor(6).view(3,2).uniform_(-5,5),requires_grad=True)

    op = QuantWeightExp(gamma=gamma, init=init, size=size)


    # Test L2
    loss1 = torch.sum(op.apply(random_tensor_1,torch.Tensor([1]),torch.Tensor([0])))
    loss1.backward()

    assert equals(
        random_tensor_1.grad,
        torch.ones_like(random_tensor_1)-exp_deriv_l2(random_tensor_1, 1, gamma, init, size)
    )


    # Test L1
    loss2 = torch.sum(op.apply(random_tensor_2,torch.Tensor([0]),torch.Tensor([1])))
    loss2.backward()

    assert equals(
        random_tensor_2.grad,
        torch.ones_like(random_tensor_2)-exp_deriv_l1(random_tensor_2, 1, gamma, init, size)
    )


