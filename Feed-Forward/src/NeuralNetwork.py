from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.input_tensor, self.label_tensor = input_tensor, label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return self.loss_layer.forward(input_tensor, label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for iteration in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        prediction = input_tensor
        for layer in self.layers:
            prediction = layer.forward(prediction)
        return prediction
