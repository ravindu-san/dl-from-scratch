import numpy as np

from Layers import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        self.input_tensor = None
        self.output_tensor = None
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        max_vec = self.input_tensor.max(axis=1, keepdims=True)
        input_tensor_exp = np.exp(self.input_tensor - max_vec)
        self.output_tensor = input_tensor_exp / np.sum(input_tensor_exp,axis=1, keepdims=True)
        return self.output_tensor

    def backward(self, error_tensor):
        return self.output_tensor * (error_tensor - np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True))
