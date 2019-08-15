from QuantTorch.layers import DorefaConv2d, LinearDorefa
from QuantTorch.functions.dorefa_connect import *
from QuantTorch.functions.dorefa_connect import _quantize
from QuantTorch.functions.common import safeSign
from random import randint
import torch
import pytest

def equals(a, b, epsilon=1e-12):
    return torch.all(torch.lt( torch.abs(a-b), epsilon ))


@pytest.mark.parametrize("inputs", [
    torch.Tensor([1,2,-1,2,3,0]).view(2,3),
    torch.Tensor([-1,-20,1,2,10]),
    torch.FloatTensor(8).view(2,2,2)
])
def test_quatize(inputs):
    "Test all _qantize feature from dorefa_connect"

    # Test with 1 bit 
    assert torch.all(torch.eq(_quantize(inputs,bit_width=1),\
                              safeSign(inputs)))

    # Test with 2-31 bit 
    for i in range(2,32,1):
        two = torch.ones_like(inputs)*2
        assert torch.all(torch.eq(_quantize(inputs,bit_width=i)  ,\
                                  ((1)/(torch.pow(two,i)-1))*torch.round((torch.pow(two,i)-1)*inputs) ))

    # Test with 32 bit 
    assert torch.all(torch.eq(_quantize(inputs,bit_width=32), inputs))


@pytest.mark.parametrize("inputs", [
    torch.FloatTensor(6).view(2,3),
    torch.FloatTensor(5).view(5,1),
    torch.FloatTensor(10).view(5,2),
])
def test_nnDorefaQuant(inputs):
    two = torch.ones_like(inputs)*2

    op_1b =  nnDorefaQuant(bit_width=1 )
    op_2b =  nnDorefaQuant(bit_width=2 )
    op_3b =  nnDorefaQuant(bit_width=3 )
    op_32b = nnDorefaQuant(bit_width=32)

    ### First check the forward step ###

    # Test with 1 bit 
    assert torch.all(torch.eq(op_1b(inputs), safeSign(inputs)))

    # Test with 2 bit 
    assert torch.all(torch.eq(op_2b(inputs)  ,\
        ((1)/(torch.pow(two,2)-1))*torch.round((torch.pow(two,2)-1)*inputs) ))

    # Test with 3 bit 
    assert torch.all(torch.eq(op_3b(inputs)  ,\
        ((1)/(torch.pow(two,3)-1))*torch.round((torch.pow(two,3)-1)*inputs) ))


    # Test with 32 bit 
    op_32b = nnDorefaQuant(bit_width=32)
    assert torch.all(torch.eq(op_32b(inputs), inputs))


    ### Then look at  the backward step ###
    """
    Here we random gen a 1xN tensor and just mul it with the inputs tensors.
    Then look at the gradiant equal to random tensors (with some dim tricks due to sum op used)
    """
    # Gen random var 
    extern_var = torch.rand(1,inputs.shape[0], requires_grad=True)

    # Test with 1 bit 
    var1b = torch.autograd.Variable(inputs, requires_grad=True)
    loss1b = torch.sum(torch.mm(extern_var,op_32b(var1b)))
    loss1b.backward()


    assert(torch.all(torch.eq(var1b.grad,
                             torch.transpose(extern_var,1,0).repeat(1,inputs.shape[-1]))))


    # Test with 2 bits 
    var2b = torch.autograd.Variable(inputs, requires_grad=True)
    loss2b = torch.sum(torch.mm(extern_var,op_2b(var2b)))
    loss2b.backward()

    assert(torch.all(torch.eq(var2b.grad,
                             torch.transpose(extern_var,1,0).repeat(1,inputs.shape[-1]))))

    # Test with 3 bits 
    var3b = torch.autograd.Variable(inputs, requires_grad=True)
    loss3b = torch.sum(torch.mm(extern_var,op_3b(var3b)))
    loss3b.backward()

    assert(torch.all(torch.eq(var3b.grad,
                             torch.transpose(extern_var,1,0).repeat(1,inputs.shape[-1]))))

    # Test with 32 bits 
    var32b = torch.autograd.Variable(inputs, requires_grad=True)
    loss32b = torch.sum(torch.mm(extern_var,op_32b(var32b)))
    loss32b.backward()

    assert(torch.all(torch.eq(var32b.grad,
                             torch.transpose(extern_var,1,0).repeat(1,inputs.shape[-1]))))


@pytest.mark.parametrize("inputs", [
    torch.FloatTensor(6).view(2,3),
    torch.FloatTensor(5).view(5,1),
    torch.FloatTensor(10).view(5,2),
])
def test_DorefaQuant(inputs):
    two = torch.ones_like(inputs)*2

    op_1b =  lambda x : DorefaQuant(x, bit_width=1 )
    op_2b =  lambda x : DorefaQuant(x, bit_width=2 )
    op_3b =  lambda x : DorefaQuant(x, bit_width=3 )
    op_32b = lambda x : DorefaQuant(x, bit_width=32 )

    ### First check the forward step ###

    # Test with 1 bit 
    assert torch.all(torch.eq(op_1b(inputs), safeSign(inputs)))

    # Test with 2 bit 
    assert torch.all(torch.eq(op_2b(inputs)  ,\
        ((1)/(torch.pow(two,2)-1))*torch.round((torch.pow(two,2)-1)*inputs) ))

    # Test with 3 bit 
    assert torch.all(torch.eq(op_3b(inputs)  ,\
        ((1)/(torch.pow(two,3)-1))*torch.round((torch.pow(two,3)-1)*inputs) ))


    # Test with 32 bit 
    op_32b = nnDorefaQuant(bit_width=32)
    assert torch.all(torch.eq(op_32b(inputs), inputs))


    ### Then look at  the backward step ###
    """
    Here we random gen a 1xN tensor and just mul it with the inputs tensors.
    Then look at the gradiant equal to random tensors (with some dim tricks due to sum op used)
    """
    # Gen random var 
    extern_var = torch.rand(1,inputs.shape[0], requires_grad=True)

    # Test with 1 bit 
    var1b = torch.autograd.Variable(inputs, requires_grad=True)
    loss1b = torch.sum(torch.mm(extern_var,op_32b(var1b)))
    loss1b.backward()


    assert(torch.all(torch.eq(var1b.grad,
                             torch.transpose(extern_var,1,0).repeat(1,inputs.shape[-1]))))


    # Test with 2 bits 
    var2b = torch.autograd.Variable(inputs, requires_grad=True)
    loss2b = torch.sum(torch.mm(extern_var,op_2b(var2b)))
    loss2b.backward()

    assert(torch.all(torch.eq(var2b.grad,
                             torch.transpose(extern_var,1,0).repeat(1,inputs.shape[-1]))))

    # Test with 3 bits 
    var3b = torch.autograd.Variable(inputs, requires_grad=True)
    loss3b = torch.sum(torch.mm(extern_var,op_3b(var3b)))
    loss3b.backward()

    assert(torch.all(torch.eq(var3b.grad,
                             torch.transpose(extern_var,1,0).repeat(1,inputs.shape[-1]))))

    # Test with 32 bits 
    var32b = torch.autograd.Variable(inputs, requires_grad=True)
    loss32b = torch.sum(torch.mm(extern_var,op_32b(var32b)))
    loss32b.backward()

    assert(torch.all(torch.eq(var32b.grad,
                             torch.transpose(extern_var,1,0).repeat(1,inputs.shape[-1]))))




@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_nnQuantWeight(weight):
    WeightQ1b  = nnQuantWeight(1 )
    WeightQ2b  = nnQuantWeight(2 )
    WeightQ3b  = nnQuantWeight(3 )
    WeightQ32b = nnQuantWeight(32)

    ### Test the forward pass ###

    # Test 1b
    assert torch.all(torch.eq(
        WeightQ1b(weight),
        safeSign(weight)* torch.mean(torch.abs(weight))
    ))

    # Test 2b
    assert torch.all(torch.eq(
        WeightQ2b(weight),
        2* _quantize( (1/2)  + torch.tanh(weight)/(2*torch.max(torch.abs(torch.tanh(weight)))) , 2) -1
    ))

    # Test 3b
    assert torch.all(torch.eq(
        WeightQ3b(weight),
        2* _quantize( (1/2)  + torch.tanh(weight)/(2*torch.max(torch.abs(torch.tanh(weight)))) , 3) -1
    ))

    # Test 32b
    assert torch.all(torch.eq(
        WeightQ32b(weight),
        weight
    ))

    ### Test the backward pass ###

    # dW/dl = 2*weight_q *  dWq/dW
    # dWq/dW = 2* d quantize(Wp)/dWp * dWp/dW
    # d quantize(Wp)/dWp = 1
    # dWp/dW = tanh(W)/2*max(abs(tanh(W))) <---- We will call that dWt

    #First we compute the dWt with torch graph backprob
    weight_var = torch.autograd.Variable(weight, requires_grad=True)

    weight_var_loss  = torch.sum(torch.tanh(weight_var)/(2*torch.max(torch.abs(torch.tanh(weight_var)))))
    weight_var_loss.backward()

    dWt = weight_var.grad


    # Test 2b
    weight_var_2b = torch.autograd.Variable(weight, requires_grad=True)
    weight_q_2b = WeightQ2b(weight_var_2b)

    loss2b = torch.sum(torch.pow(weight_q_2b, 2))
    loss2b.backward()

    assert torch.all(torch.lt(torch.abs(
            weight_var_2b.grad - 
            4*weight_q_2b*dWt),
    1e-3))

    
    # Test 3b
    weight_var_3b = torch.autograd.Variable(weight, requires_grad=True)
    weight_q_3b = WeightQ2b(weight_var_3b)

    loss3b = torch.sum(torch.pow(weight_q_3b, 2))
    loss3b.backward()

    assert torch.all(torch.lt(torch.abs(
            weight_var_3b.grad - 
            4*weight_q_3b*dWt),
    1e-3))


    # Test 32b
    weight_var_32b = torch.autograd.Variable(weight, requires_grad=True)
    weight_q_32b = WeightQ32b(weight_var_32b)

    loss32b = torch.sum(torch.pow(weight_q_32b, 2))
    loss32b.backward()

    assert torch.all(torch.eq(
            weight_var_32b.grad,
            2*weight_q_32b
    ))

def test_nnQuantWeight_div_0():
    """
    Weight quantization can have a zero div situation.
    It must don't crash and return zeros.
    """
    zeros=  torch.FloatTensor(6).uniform_(0, 0).view(3,2)
    WeightQ3b  = nnQuantWeight(3 )
    assert torch.all(torch.eq(
        zeros,
        WeightQ3b(zeros)
    ))


@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_QuantDense_without_bias(weight):
    return 
    #Generate random linear input
    rand_sample = randint(1,10)
    random_input = torch.FloatTensor(rand_sample*weight.shape[-1]).uniform_(-10,10).view(rand_sample, weight.shape[-1])
    
    op_1b  = QuantDense(1)
    op_2b  = QuantDense(2)
    op_3b  = QuantDense(3)
    op_32b = QuantDense(32)

    # Quantification op previously checked 
    weightQ1b  = nnQuantWeight(1 )
    weightQ2b  = nnQuantWeight(2 )
    weightQ3b  = nnQuantWeight(3 )
    weightQ32b = nnQuantWeight(32)


    ### Test forward pass ###

    # Test 1b
    assert torch.all(torch.eq(
        op_1b.apply(random_input, weight),
        torch.mm(random_input, weightQ1b(weight).transpose(1,0))    
    ))
    """
    #Test 2b
    assert torch.all(torch.eq(
        op_2b.apply(random_input, weight),
        torch.mm(random_input, weightQ2b(weight).transpose(1,0))    
    ))

    #Test 3b
    assert torch.all(torch.eq(
        op_3b.apply(random_input, weight),
        torch.mm(random_input, weightQ3b(weight).transpose(1,0))    
    ))
    """
    #Test 32b
    assert torch.all(torch.eq(
        op_32b.apply(random_input, weight),
        torch.mm(random_input, weightQ32b(weight).transpose(1,0))    
    ))


    ### Test backward pass ###
    # Convert all data to var data.
    random_input_var = torch.autograd.Variable(random_input, requires_grad=True)

    weight_1b = torch.autograd.Variable(weight, requires_grad=True)
    weight_2b = torch.autograd.Variable(weight, requires_grad=True)
    weight_2b_ = torch.autograd.Variable(weight, requires_grad=True)

    weight_3b = torch.autograd.Variable(weight, requires_grad=True)
    weight_3b_ = torch.autograd.Variable(weight, requires_grad=True)

    weight_32b = torch.autograd.Variable(weight, requires_grad=True)

    # Test 1 b
    loss_1b = torch.sum(op_1b.apply(random_input_var, weight_1b))
    loss_1b.backward()

    assert equals(
        weight_1b.grad,
        torch.sum(random_input_var,0).view(1,-1).repeat(weight.shape[0],1),
        1e-5
    )
    # Test 32 b
    loss_32b = torch.sum(op_32b.apply(random_input_var, weight_32b))
    loss_32b.backward()

    assert equals(
        weight_32b.grad,
        torch.sum(random_input_var,0).view(1,-1).repeat(weight.shape[0],1),
        1e-5
    )
    
    # Test 2 b
    
    loss_2b = torch.sum(op_2b.apply(random_input_var, weight_2b))
    loss_2b.backward()

    torch.sum(weightQ2b(weight_2b_)).backward()
    assert equals(
        weight_2b.grad,
        torch.sum(random_input_var,0)*weight_2b_.grad,
        1e-5
    )

    
    loss_3b = torch.sum(op_3b.apply(random_input_var, weight_3b))
    loss_3b.backward()

    torch.sum(weightQ2b(weight_3b_)).backward()

    assert equals(
        weight_3b.grad,
        torch.sum(random_input_var,0)*weight_3b_.grad,
        1e-5
    )

@pytest.mark.parametrize("weight", [
    torch.FloatTensor(6).uniform_(-10, 10).view(3,2),
    torch.FloatTensor(5).uniform_(-7, 8).view(1,5),
    torch.FloatTensor(10).uniform_(50, -10).view(2,5),
])
def test_QuantDense_with_bias(weight):
    rand_sample = randint(1,10)
    random_input_var = torch.FloatTensor(rand_sample*weight.shape[-1]).uniform_(-10,10).view(rand_sample, weight.shape[-1])
    
    


def test_QuantConv2d():
    #TODO 
    pass


if __name__ == "__main__":
    test_QuantDense_without_bias(torch.FloatTensor(6).uniform_(-10, 10).view(3,2))