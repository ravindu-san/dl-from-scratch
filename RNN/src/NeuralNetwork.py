from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

        self.input_tensor = None
        self.label_tensor = None

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.input_tensor, self.label_tensor = input_tensor, label_tensor

        regularization_loss = 0
        for layer in self.layers:
            if layer.trainable:
                regularization_loss += layer.calculate_regularization_loss()
            input_tensor = layer.forward(input_tensor)

        return self.loss_layer.forward(input_tensor, label_tensor) + regularization_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = 'train'
        for iteration in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        self.phase = 'test'
        prediction = input_tensor
        for layer in self.layers:
            layer.testing_phase = True
            prediction = layer.forward(prediction)
        return prediction
