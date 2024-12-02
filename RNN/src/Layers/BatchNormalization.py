from Layers import Base, Helpers
import numpy as np


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.weights, self.bias = self.initialize()
        self._gradient_weights = None
        self.gradient_bias = 0.
        self.iteration = 0
        self.moving_avg_decay = 0.8
        self.moving_mean = 0.
        self.moving_var = 0
        self.x_tilde = None
        self._optimizer = None
        self.batch_mean = None
        self.batch_var = None
        self.input_tensor = None
        self.input_tensor_shape = None

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

    def initialize(self, weights_initializer=None, bias_initializer=None):
        return np.ones(self.channels), np.zeros(self.channels)

    def forward(self, input_tensor):
        self.input_tensor_shape = input_tensor.shape
        self.input_tensor = input_tensor
        # if 2d image conv
        if len(self.input_tensor_shape) == 4:
            self.input_tensor = self.reformat(input_tensor)

        if self.testing_phase:
            self.x_tilde = (self.input_tensor - self.moving_mean) / (np.sqrt(self.moving_var + np.finfo(float).eps))
            output = self.weights * self.x_tilde + self.bias
        else:
            self.iteration += 1
            self.batch_mean = np.mean(self.input_tensor, axis=0)
            self.batch_var = np.var(self.input_tensor, axis=0)

            if self.iteration == 1:
                self.moving_mean = self.batch_mean
                self.moving_var = self.batch_var

            self.moving_mean = self.moving_avg_decay * self.moving_mean + (1 - self.moving_avg_decay) * self.batch_mean
            self.moving_var = self.moving_avg_decay * self.moving_var + (1 - self.moving_avg_decay) * self.batch_var

            self.x_tilde = (self.input_tensor - self.batch_mean) / (np.sqrt(self.batch_var + np.finfo(float).eps))
            output = self.weights * self.x_tilde + self.bias

        if len(self.input_tensor_shape) == 4:
            output = self.reformat(output)
        return output

    def backward(self, error_tensor):
        # if 2d image conv
        if len(self.input_tensor_shape) == 4:
            error_tensor = self.reformat(error_tensor)

        gradient_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.batch_mean,
                                                      self.batch_var)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        self._gradient_weights = np.sum(error_tensor * self.x_tilde, axis=0)

        if self.optimizer:
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        if len(self.input_tensor_shape) == 4:
            gradient_input = self.reformat(gradient_input)

        return gradient_input

    def reformat(self, tensor):
        tensor_shape = tensor.shape
        if len(tensor.shape) == 4:
            tensor = tensor.reshape(*tensor_shape[:-2], np.prod(tensor_shape[2:]))
            tensor = tensor.transpose((0, 2, 1))
            tensor = tensor.reshape(np.prod(tensor.shape[:2]), tensor.shape[-1])
        if len(tensor_shape) == 2:
            tensor = tensor.reshape(self.input_tensor_shape[0], np.prod(self.input_tensor_shape[2:]),
                                    self.input_tensor_shape[1])
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(self.input_tensor_shape[0], self.input_tensor_shape[1],
                                    *self.input_tensor_shape[2:])
        return tensor

    def calculate_regularization_loss(self):
        if self.optimizer.regularizer:
            return self.optimizer.regularizer.norm(self.weights)
        else:
            return 0
