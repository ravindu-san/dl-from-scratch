from Layers import Base
import numpy as np


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.input_tensor = None
        super().__init__()
        self.trainable = True
        self._optimizer = None
        # transposed weights
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(self.output_size, self.input_size + 1)).T
        # self.weights = None
        # self.bias = None
        self._gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    def forward(self, input_tensor):
        bias = np.ones(shape=(input_tensor.shape[0], 1))
        self.input_tensor = np.concatenate((input_tensor, bias), axis=1)
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        gradient_wrt_input = np.dot(error_tensor, self.weights.T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)  # gradient_wrt_weights
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return gradient_wrt_input[:, 0:-1]

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.output_size, self.input_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((self.output_size,1))
        self.weights = np.concatenate((weights, bias), axis=1).T


