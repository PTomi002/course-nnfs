import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # We’re initializing weights to be (inputs, neurons) instead of transposing every time we perform a forward pass.
        # Multiply by 0.01 to make them smaller.
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # The most common initialization for biases is 0.
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Forward pass: passing the input through the network.
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLuActivation:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class SoftmaxActivation:
    def __init__(self):
        self.output = None

    # Input looks like:
    #   Neuron1  2     3
    # | v1,1   v1,2   v1,3   | Input1
    # | ..............       | .....
    # | v300,1 v300,2 v300,3 | Input300
    def forward(self, inputs):
        # Unnormalized probabilities.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalized probabilities.
        # Divide the exponential values within each row with the sum of the row.
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
