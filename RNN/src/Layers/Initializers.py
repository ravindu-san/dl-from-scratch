import numpy as np
from math import sqrt


class Constant:
    def __init__(self, const_val=0.1):
        self.const_val = const_val

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.full(shape=weights_shape, fill_value=self.const_val)


class UniformRandom:
    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.random.uniform(low=0.0, high=1.0, size=weights_shape)


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        mu, sigma = 0, sqrt(2 / (fan_in + fan_out))
        return np.random.normal(mu, sigma, size=weights_shape)


class He:
    def initialize(self, weights_shape, fan_in, fan_out=None):
        mu, sigma = 0, sqrt(2 / (fan_in))
        return np.random.normal(mu, sigma, size=weights_shape)
