################
Terner Tools
################
Terner net is an extension of Binary Net. This technic add 0 value to avaliable value of BinaryNet.
This technic is explain at this `link <https://arxiv.org/pdf/1510.03009.pdf>`_ by Matthieu Courbariaux and Zhouhan Lin.
Like Binary Net, we have a deterministic and sochastic way to quantize weight. 

*************
Introduction
*************
The key idea here is to get a quantize value in the terner set (:math:`(-1,1,0)`) from floating weight.
The gradient is apply on the floating one with a straight-through estimator.

Here you can see the deterministic quantize function with it derivative. This function is not introdius on the original paper but
result directly from the original approch on the binary connect paper.


.. centered::
    forward :
.. math::
        weight_q =  \left\{
                                \begin{array}{ll}
                                +1 & \mbox{if } weight>0.5 \\
                                0 & \mbox{if }  0.5>weight>-0.5  \\
                                -1 & \mbox{if }  -0.5>weight \\
                                \end{array}
                        \right.
.. centered::
        backward :
.. math::
    \frac{\partial weight}{\partial weight_q} =  \unicode{x1D7D9}_{|{weight}| \leq 1}

Now you can see the stochastic version avaliable on this library : 

.. centered::
    forward :
.. math::
        weight_q =  \left\{
                                \begin{array}{ll}
                                +1 & \mbox{if } weight>0 \mbox{ with probability of } weight \\
                                0 & \mbox{if }  weight>0 \mbox{ with probability of } 1-weight \\
                                0 & \mbox{if }  weight<0 \mbox{ with probability of } 1+weight \\
                                -1 & \mbox{if }  weight<0 \mbox{ with probability of } -weight \\
                                \end{array}
                        \right.
.. centered::
        backward :
.. math::
    \frac{\partial weight}{\partial weight_q} =  \unicode{x1D7D9}_{|{weight}| \leq 1}




*************
API
*************




Hight level class
-------------------------

.. autoclass:: QuantTorch.layers.terner_layers.LinearTer
    :members:

    .. automethod:: __init__


.. autoclass:: QuantTorch.layers.terner_layers.TerConv2d
    :members:

    .. automethod:: __init__



Low level class
-------------------------

.. autoclass:: QuantTorch.functions.terner_connect.TernaryConnectDeterministic

.. autoclass:: QuantTorch.functions.terner_connect.TernaryConnectStochastic

.. autofunction:: QuantTorch.functions.terner_connect.TernaryConnect

.. autofunction:: QuantTorch.functions.terner_connect.TernaryDense

.. autofunction:: QuantTorch.functions.terner_connect.TernaryConv2d
