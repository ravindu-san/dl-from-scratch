from Layers import Base
import numpy as np


class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self._sigmoid = None

    @property
    def sigmoid(self):
        return self._sigmoid

    @sigmoid.setter
    def sigmoid(self, sigmoid):
        self._sigmoid = sigmoid

    def forward(self, input_tensor):
        self.sigmoid = 1 / (1 + np.exp(-input_tensor))
        return self.sigmoid

    def backward(self, error_tensor):
        return error_tensor * self.sigmoid * (1 - self.sigmoid)
