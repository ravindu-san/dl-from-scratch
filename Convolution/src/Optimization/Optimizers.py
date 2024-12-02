import numpy as np


class Sgd:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor: np.array, gradient_tensor: np.array) -> np.array:
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = -self.learning_rate * gradient_tensor
        else:
            self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.v = None
        self.r = None
        self.iteration = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None and self.r is None:
            self.v = (1-self.mu) * gradient_tensor
            self.r = (1-self.rho)* np.multiply(gradient_tensor, gradient_tensor)
            self.iteration = 1
        else:
            self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
            self.r = self.rho * self.r + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)
            self.iteration += 1

        v_hat = self.v / (1 - self.mu ** self.iteration)
        r_hat = self.r / (1 - self.rho ** self.iteration)
        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + np.finfo(float).eps)


