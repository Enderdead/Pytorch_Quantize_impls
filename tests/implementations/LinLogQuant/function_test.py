from QuantTorch.functions.log_lin_connect import LogQuant, LinQuant, nnQuant, Quant
import pytest
import torch

def equals(a, b, epsilon=1e-12):
    return torch.all(torch.lt( torch.abs(a-b), epsilon ))


@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("inputs", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_LogQuant_forward(fsr, bit_width, inputs):
    op1 =  LinQuant(fsr=fsr, bit_width=bit_width, with_sign=False, lin_back=True)
    op2 =  LinQuant(fsr=fsr, bit_width=bit_width, with_sign=True, lin_back=True)

    # Clip(Round(x/step)*step,0,2^(FSR)) with step = 2^{FSR-bitwight}
    step =  torch.FloatTensor([2]).pow(fsr-bit_width)
    expected1 = torch.clamp(torch.round(inputs/step)*step, 0,2**fsr)  
    assert equals(
        op1.apply(inputs),
        expected1
    )

    expected2 = torch.sign(inputs)*torch.clamp(torch.round(torch.abs(inputs)/step)*step, 0,2**fsr)  

    assert equals(
        op2.apply(inputs),
        expected2
    )

@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("inputs", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_LinQuant_forward(fsr, bit_width, inputs):
    op1 =  LogQuant(fsr=fsr, bit_width=bit_width, with_sign=False, lin_back=True)
 
    expected1 = torch.pow(torch.ones_like(inputs)*2, torch.clamp(torch.round(torch.log2(torch.abs(inputs))), fsr-2**bit_width ,fsr )) 

    assert equals(
        op1.apply(inputs),
        expected1
    )

    op2 =  LogQuant(fsr=fsr, bit_width=bit_width, with_sign=True, lin_back=True)
    expected2 = torch.sign(inputs)*torch.pow(torch.ones_like(inputs)*2, torch.clamp(torch.round(torch.log2(torch.abs(inputs))), fsr-2**bit_width ,fsr )) 

    assert equals(
        op2.apply(inputs),
        expected2
    )


@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("inputs", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_LinQuant_backward(fsr, bit_width, inputs):
    var_inputs_1 = torch.autograd.Variable(inputs, requires_grad=True)
    var_inputs_2 = torch.autograd.Variable(inputs, requires_grad=True)

    op =  LinQuant(fsr=fsr, bit_width=bit_width, with_sign=False, lin_back=True)

    loss1 = torch.sum(op.apply(var_inputs_1))
    loss1.backward()

    assert equals(
        var_inputs_1.grad,
        torch.ones_like(inputs)
        )

    loss2 = torch.sum(torch.pow(op.apply(var_inputs_2),2))
    loss2.backward()

    assert equals(
        var_inputs_2.grad,
        2*op.apply(inputs)
        )


@pytest.mark.parametrize("fsr",(7,6,5,4,3,2))
@pytest.mark.parametrize("bit_width",(2,3,4))
@pytest.mark.parametrize("inputs", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_LogQuant_backward(fsr, bit_width, inputs):
    var_inputs_1 = torch.autograd.Variable(inputs, requires_grad=True)
    var_inputs_2 = torch.autograd.Variable(inputs, requires_grad=True)

    op =  LogQuant(fsr=fsr, bit_width=bit_width, with_sign=False, lin_back=True)

    loss1 = torch.sum(op.apply(var_inputs_1))
    loss1.backward()

    assert equals(
        var_inputs_1.grad,
        torch.ones_like(inputs)
        )

    loss2 = torch.sum(torch.pow(op.apply(var_inputs_2),2))
    loss2.backward()

    assert equals(
        var_inputs_2.grad,
        2*op.apply(inputs)
        )

