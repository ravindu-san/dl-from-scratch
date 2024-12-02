from Layers import Base
import numpy as np


class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self._tanh = None

    @property
    def tanh(self):
        return self._tanh

    @tanh.setter
    def tanh(self, tanh):
        self._tanh = tanh

    def forward(self, input_tensor):
        self.tanh = np.tanh(input_tensor)
        return self.tanh

    def backward(self, error_tensor):
        return error_tensor * (1 - np.square(self.tanh))
