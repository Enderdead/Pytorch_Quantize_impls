import random
from scipy.stats import uniform
from math import log10

class Parameter:
    def __init__(self, name):
        self.name = name

    def apply(self, trial=None):
        raise NotImplementedError

class UniformFloat(Parameter):
    def __init__(self, name, min, max):
        Parameter.__init__(self, name)
        self.generator = uniform(min, max)
        self.params = [min, max]

    def apply(self, trial=None):
        if trial is None:
            return self.generator.rvs(1)[0]
        return trial.suggest_uniform(self.name, *self.params)


class UniformInt(Parameter):
    def __init__(self, name, min, max):
        Parameter.__init__(self, name)
        self.generator = uniform(min, max)
        self.params = [min, max]

    def apply(self, trial=None):
        if trial is None:
            return int(self.generator.rvs(1)[0])
        return trial.suggest_int(self.name, *self.params)


class UniformLog(Parameter):
    def __init__(self, name, min, max):
        Parameter.__init__(self, name)
        self.params = [min, max]
        self.generator = uniform(log10(min), log10(max))

    def apply(self, trial=None):
        if trial is None:
            return 10**(self.generator.rvs(1)[0])
        return trial.suggest_loguniform(self.name, *self.params)

class DiscretParameter(Parameter):
    def __init__(self, name, *values):
        Parameter.__init__(self, name)
        self.values = values

    def apply(self, trial=None):
        if trial is None:
            return random.choice(self.values)
        return trial.suggest_categorical(self.name, self.values)
