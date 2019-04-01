import random

class Parameter:
    def __init__(self, name):
        self.name = name

    def apply(self, trial=None):
        raise NotImplementedError

class UniformFloat(Parameter):
    def __init__(self, name, min, max):
        Parameter.__init__(self, name)
        self.params = [min, max]

    def apply(self, trial=None):
        return trial.suggest_uniform(self.name, *self.params)


class UniformInt(Parameter):
    def __init__(self, name, min, max):
        Parameter.__init__(self, name)
        self.params = [min, max]

    def apply(self, trial=None):
        return trial.suggest_int(self.name, *self.params)


class UniformLog(Parameter):
    def __init__(self, name, min, max):
        Parameter.__init__(self, name)
        self.params = [min, max]

    def apply(self, trial=None):
        return trial.suggest_loguniform(self.name, *self.params)

class DiscretParameter(Parameter):
    def __init__(self, name, *values):
        Parameter.__init__(self, name)
        self.values = values

    def apply(self, trial=None):
        return trial.suggest_discrete_uniform(self.name, *self.values)
