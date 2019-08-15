from .binary_connect import BinaryConnectDeterministic,\
                            BinaryConnectStochastic,\
                            BinaryConnect,\
                            BinaryDense,\
                            BinaryConv2d,\
                            AP2,\
                            ShiftBatch

from .dorefa_connect import nnDorefaQuant,\
                            DorefaQuant,\
                            nnQuantWeight,\
                            QuantDense,\
                            QuantConv2d

from .elastic_quant_connect import lin_proj,\
                            exp_proj,\
                            lin_deriv_l2,\
                            exp_deriv_l2,\
                            lin_deriv_l1,\
                            exp_deriv_l1,\
                            QuantWeightLin,\
                            QuantWeightExp,\
                            QuantLinDense,\
                            QuantLogDense,\
                            QuantConv2d


from .log_lin_connect import    LogQuant,\
                                LinQuant,\
                                nnQuant,\
                                Quant

from .terner_connect import TernaryConnectDeterministic,\
                            TernaryConnectStochastic,\
                            TernaryConnect,\
                            TernaryDense,\
                            TernaryConv2d

from .xnor_connect import   nnQuantXnor,\
                            QuantXnor,\
                            XNORDense,\
                            XNORConv2d

from .WQR_connect import    lin_deriv_WQR,\
                            exp_deriv_WQR,\
                            QuantWeightWLin,\
                            QuantWeightWExp,\
                            QuantWLinDense,\
                            QuantWLogDense

from .common import safeSign