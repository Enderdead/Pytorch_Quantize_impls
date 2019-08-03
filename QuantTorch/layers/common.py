class QLayer():
    def get_quant_weight(self):
        raise NotImplementedError
    
    def set_quant_weight(self):
        raise NotImplementedError
    
    def restore_weight(self):
        raise NotImplementedError