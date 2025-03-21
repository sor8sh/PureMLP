import numpy as np
from numpy import ndarray


class Funcs:
    @staticmethod
    def relu(X: ndarray) -> ndarray:
        return np.maximum(0, X)

    @staticmethod
    def d_relu(X: ndarray) -> ndarray:
        return (X > 0).astype(float)

    @staticmethod
    def sigmoid(X: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def d_sigmoid(X: ndarray) -> ndarray:
        return X * (1 - X)

    @staticmethod
    def tanh(X: ndarray) -> ndarray:
        return np.tanh(X)

    @staticmethod
    def d_tanh(X: ndarray) -> ndarray:
        return 1 - X ** 2

    @staticmethod
    def softmax(X: ndarray) -> ndarray:
        exp = np.exp(X - np.max(X, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    @staticmethod
    def d_softmax(X: ndarray) -> ndarray:
        return X * (1 - X)
