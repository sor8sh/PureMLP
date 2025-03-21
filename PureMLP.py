import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from numpy import ndarray
from Utils import Utils


class MLP:
    """
    A minimal Multilayer Perceptron (feedforward neural network) class
    with customizable layer sizes and activation functions
    """

    def __init__(self, layers: List[int], activation_functions: List[str]) -> None:
        """
        - layers: List defining number of neurons in each layer, e.g., [784, 64, 10]
        - activation_functions: List of activation function names for each layer except input, e.g., ['relu', 'softmax']
        """
        self.layers = layers
        self.f_activations = activation_functions
        self.num_layers = len(layers) - 1  # Number of layers excluding input layer
        self.params = Utils.init_params(layers)
        self.cache = {}  # Store intermediate values from forward propagation for backpropagation

    def forward(self, X: ndarray) -> ndarray:
        """
        Perform forward propagation through the network
        - X: Input data of shape (features, samples)
        => Output activations of the final layer
        """
        self.cache, A = {'A0': X}, X
        for i in range(1, self.num_layers + 1):
            Z = self.params[f'W{i}'].dot(A) + self.params[f'b{i}']
            A = Utils.activation(Z, self.f_activations[i - 1])
            self.cache[f'Z{i}'], self.cache[f'A{i}'] = Z, A
        return A

    def backward(self, X: ndarray, Y: ndarray) -> Dict:
        """
        Perform backpropagation and compute gradients
        - X: Input data of shape (features, samples)
        - Y: Target values of shape (output_size, samples)
        => Dictionary containing gradients for all weights and biases
        """
        m, grads = X.shape[1], {}
        dA = self.cache[f'A{self.num_layers}'] - Y  # Loss gradient (Softmax with Cross-Entropy)
        for i in range(self.num_layers, 0, -1):
            dZ = dA * Utils.d_activation(self.cache[f'A{i}'], self.cache[f'Z{i}'], self.f_activations[i - 1])
            grads[f'dW{i}'] = (1 / m) * dZ.dot(self.cache[f'A{i - 1}'].T)
            grads[f'db{i}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = self.params[f'W{i}'].T.dot(dZ)  # Propagate error backward
        return grads

    def update_params(self, grads: Dict, learning_rate: float) -> None:
        """
        Update weights and biases using gradient descent
        - grads: Dictionary containing gradients for all weights and biases
        - learning_rate: Learning rate for gradient descent
        """
        for i in range(1, self.num_layers + 1):
            self.params[f'W{i}'] -= learning_rate * grads[f'dW{i}']
            self.params[f'b{i}'] -= learning_rate * grads[f'db{i}']

    def train(self, X: ndarray, Y: ndarray,
              epochs: int = 5, learning_rate: float = 0.01,
              batch_size: int = 64, verbose: int = 1) -> None:
        """
        Train the network using mini-batch gradient descent
        - X: Training data of shape (features, samples)
        - Y: Training labels of shape (1, samples) or (samples,)
        - epochs: Number of training epochs. Defaults to 5
        - learning_rate: Learning rate for gradient descent. Defaults to 0.01
        - batch_size: Size of mini-batches. Defaults to 64
        - verbose: Print progress every 'verbose' epochs. If 0, no output. Defaults to 1
        """
        m, Y = X.shape[1], Utils.one_hot(Y, self.layers[-1])
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled, Y_shuffled = X[:, indices], Y[:, indices]
            for j in range(0, m, batch_size):
                X_batch, Y_batch = X_shuffled[:, j:j + batch_size], Y_shuffled[:, j:j + batch_size]
                A = self.forward(X_batch)
                grads = self.backward(X_batch, Y_batch)
                self.update_params(grads, learning_rate)
            if verbose and epoch % verbose == 0: Utils.log(epoch, A, Y_batch)

    def predict(self, X: ndarray) -> ndarray:
        """
        Predict label using the trained model
        - X: Input data of shape (features, samples)
        => Predicted class labels
        """
        A = self.forward(X)
        return Utils.predictions(A)

    def test(self, X: ndarray, Y: int) -> None:
        """
        Test the network on a single image and visualize the result
        - X: Input image of shape (features, 1)
        - Y: True label for the image
        """
        print(f"Prediction: {self.predict(X)}")
        print(f"Label: {Y}")
        plt.gray()
        plt.imshow(X.reshape((28, 28)) * 255, interpolation='nearest')
        plt.show()
