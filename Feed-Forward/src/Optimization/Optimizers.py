import numpy as np


class Sgd:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor: np.array, gradient_tensor: np.array) -> np.array:
        return weight_tensor - self.learning_rate * gradient_tensor
