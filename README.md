# Multilayer Perceptron with NumPy

`PureMLP` is a minimal, customizable feed-forward neural network implemented using only NumPy.
It supports configurable layer sizes, activation functions, forward propagation, backpropagation, and mini-batch training with gradient descent.
It also includes a basic testing and visualization function for classification tasks.

## Requirements
- `NumPy`
- `Pandas`
- `Matplotlib`

## Usage

### Initialize the Network
```python
from PureMLP import MLP

model = MLP(layers=[784, 64, 10], activation_functions=['relu', 'softmax'])
```

### Train
```python
model.train(X_train, Y_train, epochs=10, learning_rate=0.01, batch_size=64)
```

### Predict
```python
predictions = model.predict(X_test)
```

### Test with Visualization
```python
model.test(X_sample, true_label)
```
