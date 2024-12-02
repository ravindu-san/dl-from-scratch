from Layers import Base
import numpy as np
from Layers import FullyConnected, TanH, Sigmoid


class RNN(Base.BaseLayer):
    def __init__(self, input_size: int, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trainable = True
        self.memorize = False
        self.fc_layer_1 = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc_layer_2 = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        # self.weights_hy = self.fc_layer_2.weights
        self.tanh = TanH.TanH()
        self.sigmoid = Sigmoid.Sigmoid()
        self.hidden_states = []
        self.sigmoid_activations = []
        self.tanh_activations = []
        self.gradient_hidden_states = []
        self._gradient_weights = 0
        self._gradient_weights_out = 0
        self._optimizer = None
        self.input_tensors_fc_hidden = []
        self.input_tensors_fc_out = []

    @property
    def weights(self):
        return self.fc_layer_1.weights

    @weights.setter
    def weights(self, weights):
        self.fc_layer_1.weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        self.tanh_activations = []
        self.sigmoid_activations = []
        self.input_tensors_fc_hidden = []
        self.input_tensors_fc_out = []
        self.gradient_weights = 0

        # use memorize boolean if next sequence is also a part of current
        if (not self.memorize) or (self.memorize and not self.hidden_states):
            self.hidden_states = [np.zeros((1, self.hidden_size))]

        output = []
        for t in range(input_tensor.shape[0]):
            input_vector = input_tensor[t]
            x_hidden_concat = np.concatenate((input_vector.reshape(1, *input_vector.shape), self.hidden_states[-1]), axis=1)
            fc_hidden = self.fc_layer_1.forward(x_hidden_concat)
            self.input_tensors_fc_hidden.append(self.fc_layer_1.input_tensor)
            hidden_state = self.tanh.forward(fc_hidden)
            self.tanh_activations.append(hidden_state)
            self.hidden_states.append(hidden_state)

            fc_out = self.fc_layer_2.forward(hidden_state)
            self.input_tensors_fc_out.append(self.fc_layer_2.input_tensor)
            sigmoid_activation = self.sigmoid.forward(fc_out)
            self.sigmoid_activations.append(sigmoid_activation)
            output.append(sigmoid_activation)
        return np.squeeze(np.array(output), axis=1)

    def backward(self, error_tensor):
        if (not self.memorize) or (self.memorize and not self.gradient_hidden_states):
            self.gradient_hidden_states = [np.zeros((1, self.hidden_size))]

        output = []
        for t in reversed(range(error_tensor.shape[0])):
            error = error_tensor[t]
            self.sigmoid.sigmoid = self.sigmoid_activations[t]
            gradient_sigmoid = self.sigmoid.backward(error.reshape(1, *error.shape))
            self.fc_layer_2.input_tensor = self.input_tensors_fc_out[t]
            gradient_fc_out = self.fc_layer_2.backward(gradient_sigmoid)
            self._gradient_weights_out += self.fc_layer_2.gradient_weights

            gradient_cpy = gradient_fc_out + self.gradient_hidden_states[-1]

            self.tanh.tanh = self.tanh_activations[t]
            gradient_tanh = self.tanh.backward(gradient_cpy)
            self.fc_layer_1.input_tensor = self.input_tensors_fc_hidden[t]
            gradient_fc_hidden = self.fc_layer_1.backward(gradient_tanh)
            split = np.split(gradient_fc_hidden, (self.input_size,), axis=1)
            self.gradient_hidden_states.append(split[1])
            output.append(split[0])
            self.gradient_weights += self.fc_layer_1.gradient_weights

        if self.optimizer:
            self.fc_layer_1.weights = self.optimizer.calculate_update(self.fc_layer_1.weights, self.gradient_weights)
            self.fc_layer_2.weights = self.optimizer.calculate_update(self.fc_layer_2.weights,
                                                                      self._gradient_weights_out)
        return np.squeeze(np.array(output[::-1]), axis=1)

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_layer_1.initialize(weights_initializer, bias_initializer)
        self.fc_layer_2.initialize(weights_initializer, bias_initializer)
        # self.weights = self.fc_layer_1.weights

    def calculate_regularization_loss(self):
        if self.optimizer.regularizer:
            fc_hidden_loss = self.optimizer.regularizer.norm(self.weights)
            fc_out_loss = self.optimizer.regularizer.norm(self.fc_layer_2.weights)
            return fc_hidden_loss + fc_out_loss
        else:
            return 0

