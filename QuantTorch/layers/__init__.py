from .binary_layers import  LinearBin,\
                            BinConv2d,\
                            ShiftNormBatch1d,\
                            ShiftNormBatch2d

from .dorefa_layers import  LinearDorefa,\
                            DorefaConv2d
                         

from .elastic_layers import LinearQuantLin,\
                            LinearQuantLog,\
                            QuantConv2dLin,\
                            QuantConv2dLog


from .log_lin_layers import  LinearQuant,\
                             QuantConv2d

from .terner_layers import LinearTer,\
                            TerConv2d

from .xnor_layers import   LinearXNOR,\
                            XNORConv2d

from .WQR_layers import    LinearQuantWLin,\
                            LinearQuantWLog,\
                            QuantConv2dWLin,\
                            QuantConv2dWLog