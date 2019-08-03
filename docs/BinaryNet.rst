################
Binary-net tools
################
Binary-net is a binarization strategy introduced by Matthieu Courbariaux and Itay Hubara in 2016.
The main idea is to use the sign op to binarize weight to -1 or 1. The sign back propagation uses a straight-through estimator.


*************
Introduction
*************

Binary net deal with weight and activation in order to get a very efficientness computation on matrix multiplication. 
The weight transformation can be perform with stochastic or deterministic process. On deterministic process, we just apply the 
`sign` op and we use the identity straight-through. 

	

.. centered::
    forward :
.. math::
    weight_{bin} = sign(weight)
.. centered::
        backward :
.. math::
    \frac{\partial weight_{bin}}{\partial weight} \equiv \unicode{x1D7D9}_{|{weight}| \leq 1}

On stochastic process, the sign operator is replace with a random variable following a  probability given by the weight.



.. centered::
    forward :
.. math::
        weight_{i, j}=  \left\{
                                \begin{array}{ll}
                                +1 & \mbox{with probability } p= \sigma (weight_{i, j}) \\
                                -1 & \mbox{with probability }  1 -p \\
                                \end{array}
                        \right.
.. centered::
    backward :
.. math::
        \frac{\partial weight_{bin}}{\partial weight} \equiv \unicode{x1D7D9}_{|{weight}| \leq 1}

The stochastic weight binarization can be use to normalize your model and replace some dropout layers.
The input layer binarization is usually compute with the deterministic process.



*************
API
*************


Hight level class
-------------------------

.. class:: LinearBin(torch.nn.Module)

    .. method:: __init__(in_features, out_features, bias=True)

        The basic LinearBin constructor with layer sizing.

        :param in_features:
            Inputs features size.
        :param out_features:
            Output features size.
        :param bool bias:
            Use a bias on linear application or not.

    .. method:: reset_parameters()
        
        Reset weight values with random normal distribution.

    .. method:: clamp()
        
        clamp weight values between -1 and 1.

    .. method:: forward(input)
       
        Apply the layer on input tensor.



.. autoclass:: QuantTorch.layers.binary_layers.BinConv2d

    .. automethod:: __init__


Low level class
-------------------------

.. autoclass:: QuantTorch.functions.binary_connect.BinaryConnectDeterministic
    :members:

.. autoclass:: QuantTorch.functions.binary_connect.BinaryConnectStochastic
    :members:

.. autofunction:: QuantTorch.functions.binary_connect.BinaryConnect

.. autoclass:: QuantTorch.functions.binary_connect.BinaryDense

.. autofunction:: QuantTorch.functions.binary_connect.AP2

.. autoclass:: QuantTorch.functions.binary_connect.ShiftBatch

.. autofunction:: QuantTorch.functions.binary_connect.BinaryConv2d
