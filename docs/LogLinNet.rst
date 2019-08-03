################
Log Lin tools
################

Log Lin Net refer to a quantization technic introduced on this `paper <https://arxiv.org/pdf/1603.01025.pdf>`_ by a Daisuke Miyashita.
This technic is very similar to Binary net, we have a floating weight version, used to compute the gradient descent algorithm. Log Lin Tools have
two way to compute the quantize weight version. The logarithm and Linear strategies, the first one use only power of 2 numbers and the second use a restricted set of values.

*************
Introduction
*************
To compute the quantization, the lin and log function needs hyperparameters; fsr and bitwidth.
FSR is an offset used to deals with differents weight range. This parameter represent the max value of the quantized result set.

.. centered::
    The logarithm version is defined as follow : 

.. math::
    LinearQuant(x, bitwidth, FSR) = Clip(Round(\frac{x}{step} \cdot step, 0, 2^{FSR})
.. math::
    \text{with } step = 2^{FSR-bitwidth}

.. centered::
    The Linear version is defined as follow : 

.. math::
    LogQuant(x, bitwidth, FSR) =\left\{
    \begin{array}{ll}
	  0 &  si\;\; \; x=0 \\
          2^{\widetilde x } & sinon 
    \end{array}
    \right. 

.. math::
    \text{with }\widetilde x = Clip(Round(log_2(|x|),FSR-2^{bitwidht}, FSR) 


All theres function use identity traight-through estimator ( :math:`\frac{\partial x_q}{\partial x} =1`).

Weights and activations can be quantized with the same way but Weights need to be initialized on good range of values (in order to don't be satured on the quantize set).


*************
API
*************

Hight level class
-------------------------

.. autoclass:: QuantTorch.layers.log_lin_layers.LinearQuant
    :members:

.. autoclass:: QuantTorch.layers.log_lin_layers.QuantConv2d
    :members:

Low level class
-------------------------

.. autofunction:: QuantTorch.functions.log_lin_connect.LogQuant

.. autofunction:: QuantTorch.functions.log_lin_connect.LinQuant

.. autofunction:: QuantTorch.functions.log_lin_connect.nnQuant

.. autofunction:: QuantTorch.functions.log_lin_connect.Quant
