from Layers import Base
import numpy as np


class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        self.mask = np.random.rand(*input_tensor.shape) < (1 - self.probability)
        input_tensor[self.mask] = 0.0
        return input_tensor / self.probability

    def backward(self, error_tensor):
        if self.testing_phase:
            return error_tensor
        error_tensor[self.mask] = 0.0
        return error_tensor / self.probability
