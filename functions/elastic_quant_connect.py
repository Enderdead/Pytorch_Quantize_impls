import torch
from functions.common import front
from device import device
import warnings
warnings.simplefilter("always",DeprecationWarning)


def _proj_val(x, set):
    """
    Compute the projection from x onto the set given.

    :param x: Input pytorch tensor.
    :param set: Input pytorch vector used to perform the projection.
    """
    x = x.repeat((set.size()[0],)+(1,)*len(x.size()))
    x = x.permute(*(tuple(range(len(x.size())))[1:]  +(0,) ))
    x = torch.abs(x-set)
    x = torch.argmin(x, dim=len(x.size())-1, keepdim=False)
    return set[x]

def lin_proj(x, top=1, bottom=-1, size=5):
    step  = (top-bottom)/size
    lin_set = torch.arange(bottom, top+step,step=step).to(x.device)
    return _proj_val(x, lin_set)

def exp_proj(x, gamma=2, init=0.25, size=5):
    exp_set = torch.ones(size*2).to(x.device)
    for index in range(size):
        exp_set[size-1-index] = init*(gamma**index)
        exp_set[size+index]   = init*(gamma**index)
    return _proj_val(x, exp_set)

def lin_deriv_l2(x, alpha, top=1,  bottom=-1, size=5):
    """
    Apply a Sawtooth function on x with alpha as coefficient. This function is null on size specific values.
    There's values start from bottom and finish at top, and they have a  unifom  step between each other.

    :param x: tensor used to return y.
    :param alpha: coef for the Sawtooth function
    :param bottom: Lower value with null output.
    :param top: Upper value with null output.
    :param size: Number of values with null output on this function.

    Here an example of this function with size = 2 and alpha = 1. 

    ..
         y
        /\                 +_
         |                   +_
         |                     +_
         |                       +_
         |                         +_
         ---------+--------+---------+------> x
           bottom  +_                 top
                     +_      
                       +_   
                         +_ 
                           +           
                                   
    """
    delta = (top-bottom)/(size-1)
    res = torch.zeros_like(x)
    for i in range(size):
        res += (-alpha*x +(bottom+i*delta)*alpha)*(x<(bottom+(i)*delta+delta/2)).float()*(x>(bottom+(i)*delta-delta/2)).float()
    return res


def exp_deriv_l2(x, alpha, gamma=2, init=0.25, size=5):
    r"""
    Apply a Sawtooth function on x with alpha as coefficient. This function is null on size specific values.
    Roots values are computed with a geometrical sequence with init value = [init, -init] and scale factor = gamma.  
        
    :param x: tensor used to return y.
    :param alpha : coef for the Sawtooth function
    :param bottom: Lower value with null output.
    :param top   : Upper value with null output.
    :param size  : Number of values with null output on this function.

    Example of Root values : 

    root_x = [-init, +init]

    root_x = [-init*gamma, -init, +init, +init*gamma]

    root_x = [-init*gamma**2, -init*gamma, -init, +init, +init*gamma, +init*gamma**2]

        .

        .

        .  
                           
    """
    res = torch.zeros_like(x)
    res+= -alpha*(x - init) *(x > 0).float()* (x < (init*gamma + init) / 2).float()
    res+= -alpha*(x + init) *(x < 0).float()* (x > (-init*gamma + -init) / 2).float()
    cur = init
    for _ in range(size-1):
        previous = cur
        cur *=gamma
        res += -alpha * (x - cur)* (x > (cur + previous) / 2).float() *(x < (cur + gamma*cur)/2).float()
        res +=  -alpha *(x + cur)* (x < (-cur +- previous) / 2).float() *(x > (-cur + -gamma*cur)/2).float()
    
    return res


def lin_deriv_l1(x,beta,top=1, bottom=-1, size=5):
    delta = (top-bottom)/(size-1)
    res = torch.zeros_like(x)
    for i in range(size):
        res += beta*(x<(bottom+(i)*delta+delta/2)).float()*(x>(bottom+(i)*delta)).float()
        res -= beta*(x<(bottom+(i)*delta)).float()*(x>(bottom+(i)*delta-delta/2)).float()
    return res


def exp_deriv_l1(x, beta, gamma=2, init=0.25/2, size=5):
    res = torch.zeros_like(x)
    res -= beta*(x>0).float()*(x<(init)).float()
    res += beta*(x>init).float()*(x< ((init*gamma + init)/2)).float()
    res += beta*(x<0).float()*(x>(-init)).float()
    res -= beta*(x<-init).float()*(x> ((-init*gamma - init)/2)).float()
    cur = init
    for _ in range(size-1):
        previous = cur
        cur *=gamma
        res -= beta* (x > (cur + previous) / 2).float()*(x < cur).float()
        res += beta*(x > cur).float()*(x < (cur+cur*gamma)/2).float()
        res += beta* (x < -((+cur + previous) / 2)).float()*(x > -cur).float()
        res -= beta*(x < -cur).float()*(x> (-cur-cur*gamma)/2).float()
    return res


def QuantWeightLin(top=1,  bottom=-1, size=5):
    class _QuantWeightOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, alpha, beta):
            ctx.save_for_backward(weight, alpha, beta)
            return weight
        
        @staticmethod
        def backward(ctx, output_grad):
            weight, alpha, beta = ctx.saved_variables
            input_grad  = output_grad.clone()
            input_grad -= lin_deriv_l2(weight, alpha,  top,  bottom, size)
            input_grad -= lin_deriv_l1(weight, beta,  top,  bottom, size)
            return input_grad, None, None
    return _QuantWeightOp

def QuantWeightExp(gamma=2, init=0.25, size=5):
    class _QuantWeightOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, alpha, beta):
            ctx.save_for_backward(weight, alpha, beta)
            return weight
        
        @staticmethod
        def backward(ctx, output_grad):
            weight, alpha, beta = ctx.saved_variables
            input_grad  = output_grad.clone()
            input_grad -= lin_deriv_l2(weight, alpha, gamma, init, size)
            input_grad -= lin_deriv_l1(weight, beta, gamma, init, size)

            return input_grad, None
    return _QuantWeightOp

def QuantLinDense(size=5, bottom=-1, top=1):
    """
    Return a linear transformation op with this form: y=W.x+b
    This operation inclue a backprob penality using Linear Quant method.
    """
    class _QuantLinDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias, alpha, beta):
            alpha.to(device)
            beta.to(device)
            ctx.save_for_backward(input, weight, bias, alpha, beta)
            return torch.nn.functional.linear(input, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, alpha, beta = ctx.saved_variables
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)

            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
                grad_weight -= lin_deriv_l2(weight, alpha=alpha, bottom=bottom, top=top)
                grad_weight -= lin_deriv_l1(weight, beta=beta, bottom=bottom, top=top)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
                grad_bias -= lin_deriv_l2(bias, alpha=alpha, bottom=bottom, top=top)
                grad_bias -= lin_deriv_l1(bias, beta=beta, bottom=bottom, top=top)

            return grad_input, grad_weight, grad_bias, None, None
    return _QuantLinDense


def QuantLogDense(gamma=2, init=0.25, size=5):
    """
        Return a quantization op with Log method.
        Quantized values are computed using geometrical sequence with init value = [init, -init] and scale factor = gamma.  

        param: 
            gamma
    """
    class _QuantLogDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias, alpha, beta):
            alpha.to(device)
            beta.to(device)
            ctx.save_for_backward(input, weight, bias, alpha, beta)
            return torch.nn.functional.linear(input, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, alpha, beta = ctx.saved_variables
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)

            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
                grad_weight -= exp_deriv_l2(weight, alpha=alpha, gamma=2, init=0.25, size=5)
                grad_weight -= exp_deriv_l1(weight, beta=beta, gamma=2, init=0.25, size=5)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
                grad_bias -= exp_deriv_l2(bias, alpha=alpha, gamma=2, init=0.25, size=5)
                grad_bias -= exp_deriv_l1(bias, beta=beta, gamma=2, init=0.25, size=5)

            return grad_input, grad_weight, grad_bias, None, None
    return _QuantLogDense





def QuantConv2d(size=5, bottom=-1, top=1, stride=1, padding=1, dilation=1, groups=1):
    """
        **DEPRECATED**
        Return a Conv op with params given. Use Loss quantization to quantize weight before apply it.
    """
    warnings.warn("Deprecated conv op ! Huge cuda memory consumption due to torch.grad.cuda_grad.conv2d_input function.", DeprecationWarning,stacklevel=2)
    class _QuantConv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias, alpha, beta):
            ctx.save_for_backward(input, weight, bias, alpha, beta)
            output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, alpha, beta = ctx.saved_variables

            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(input.size(), weight, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
                grad_weight -= lin_deriv_l2(weight, alpha, top, bottom, size)
                grad_weight -= lin_deriv_l1(weight, beta, top, bottom, size)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0).sum(1).squeeze(1).sum(-1).squeeze(-1)
                grad_bias -= lin_deriv_l2(bias, alpha, top, bottom, size)
                grad_bias -= lin_deriv_l1(bias, beta, top, bottom, size)

            return grad_input, grad_weight, grad_bias, None, None

    return _QuantConv2d
