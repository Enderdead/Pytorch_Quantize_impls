################
Dorefa tools
################
Dorefa net is a quantization technics on layer's weight. This method can choose any bits sizing per float. 
This strategy was introduced by a Megvii Technology team.  Paper is avaliable `here <https://arxiv.org/pdf/1606.06160.pdf>`_.
Their idea is to drop all weight on small interval (:math:`[0,1]`) before apply the quantization function.


*************
Introduction
*************
The Dorefa net quantize weights, activations and gradients. Differents strategies are apply on each case, but they always use the quantize function.
This function transform a real number into quantized number from  :math:`[0,1]` to :math:`[0,1]`.

This function is defined as follow (with k equals to bits sizing): 

.. centered::
    forward :
.. math::
        quantize_{k}(x) = \frac{1}{2^{k}-1} round((2^{k}-1)x) 
.. centered::
    backward :
.. math::
        \frac{\partial quantize_{k}(x) }{\partial x} = 1


Activations are directly quantized with the quantize function due to a restricted selection of activations functions on previous layers (like sigmoid or cliped relu).
Weight are compressed with a regularization function defined here : 

.. centered::
    Weights regularization function :

.. math::
    W_q = 2*quantize_{k}(\frac{tanh(W)}{2.max(|tanh(W)|)}+\frac{1}{2} )-1  

In this project we don't care about learning optimization so the gradients quantization is omited.


*************
API
*************




Hight level class
-------------------------

.. autoclass:: QuantTorch.layers.dorefa_layers.LinearDorefa
    :members:

    .. automethod:: __init__


.. autoclass:: QuantTorch.layers.dorefa_layers.DorefaConv2d
    :members:

    .. automethod:: __init__



Low level class
-------------------------

.. autofunction:: QuantTorch.functions.dorefa_connect._quantize

.. autofunction:: QuantTorch.functions.dorefa_connect.nnDorefaQuant

.. autofunction:: QuantTorch.functions.dorefa_connect.DorefaQuant

.. autofunction:: QuantTorch.functions.dorefa_connect.nnQuantWeight

.. autofunction:: QuantTorch.functions.dorefa_connect.QuantDense

.. autofunction:: QuantTorch.functions.dorefa_connect.QuantConv2d