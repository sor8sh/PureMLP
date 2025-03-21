import numpy as np
from typing import List, Dict, Union, Any
from numpy import ndarray
from Funcs import Funcs


class Utils:
    """
    A utility class for neural network operations
    """

    @staticmethod
    def activation(Z: ndarray, func: str) -> ndarray:
        """
        Apply activation function
        - Z: Input data (the weighted sum of inputs)
        - func: Activation function name ('relu', 'sigmoid', 'tanh', or 'softmax')
        => Output after applying the activation function
        """
        func = func.lower()
        if func == 'relu': return Funcs.relu(Z)
        elif func == 'sigmoid': return Funcs.sigmoid(Z)
        elif func == 'tanh': return Funcs.tanh(Z)
        elif func == 'softmax': return Funcs.softmax(Z)
        else: raise ValueError(f"Unsupported activation function: {func}")

    @staticmethod
    def d_activation(A: ndarray, Z: ndarray, func: str) -> Union[int, Any]:
        """
        Compute the derivative of the activation function
        - A: Output of the activation function
        - Z: Input to the activation function
        - func: Activation function name ('relu', 'sigmoid', 'tanh', or 'softmax')
        => Derivative of the activation function
        """
        if func == 'relu': return Funcs.d_relu(Z)
        elif func == 'sigmoid': return Funcs.d_sigmoid(A)
        elif func == 'tanh': return Funcs.d_tanh(A)
        elif func == 'softmax': return 1 # Softmax derivative is handled separately in loss
        else: raise ValueError(f"Unsupported activation function: {func}")

    @staticmethod
    def init_params(layers: List[int]) -> Dict:
        """
        Initialize weights and biases using Xavier initialization
        - layers: List of layer sizes (number of neurons in each layer)
        => Dictionary containing initialized weights and biases
        """
        params = {}
        for i in range(1, len(layers)):
            params[f'W{i}'] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(1 / layers[i-1])
            params[f'b{i}'] = np.zeros((layers[i], 1))
        return params

    @staticmethod
    def one_hot(Y: ndarray, num_classes: int) -> ndarray:
        """
        Convert labels to one-hot encoding
        - Y: Integer array of class labels
        - num_classes: Total number of classes
        => One-hot encoded labels with shape (num_classes, num_samples)
        """
        one_hot_Y = np.zeros((num_classes, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y

    @staticmethod
    def predictions(Output: ndarray) -> ndarray:
        """
        Predict the class with the highest probability
        - Output: Network output probabilities with shape (num_classes, num_samples)
        => Predicted class indices
        """
        return np.argmax(Output, 0)

    @staticmethod
    def accuracy(predictions: ndarray, Y: ndarray) -> ndarray:
        """
        Calculate accuracy for both one-hot encoded and raw labels
        - predictions: Predicted class indices
        - Y: True labels (one-hot encoded or class indices)
        => Classification accuracy as a value between 0 and 1
        """
        true_labels = np.argmax(Y, axis=0) if Y.ndim > 1 else Y
        return np.mean(predictions == true_labels)

    @staticmethod
    def log(epoch: int, output: ndarray, Y: ndarray) -> None:
        """
        Log the accuracy of the model
        - epoch: Current training epoch
        - Output: Network output probabilities
        - Y: True labels (one-hot encoded or class indices)
        """
        predictions = Utils.predictions(output)
        accuracy = Utils.accuracy(predictions, Y)
        print(f"Epoch {epoch}: Accuracy {accuracy}")
