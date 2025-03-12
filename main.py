from abc import abstractmethod

import numpy as np


class DenseLayer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.inputs = None
        self.output = None
        self.d_weights = None
        self.d_inputs = None
        self.d_biases = None
        # We’re initializing weights to be (inputs, neurons) instead of transposing every time we perform a forward pass.
        # Multiply by 0.01 to make them smaller.
        self.weights = 0.01 * np.random.randn(number_of_inputs, number_of_neurons)
        # The most common initialization for biases is 0.
        self.biases = np.zeros((1, number_of_neurons))

    # Forward pass: passing the input through the network.
    def forward(self, inputs):
        # Remember the inputs during backward pass
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        self.d_inputs = np.dot(d_values, self.weights)
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)


class ReLuActivation:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.d_inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        # Activating only the relevant paths of the neurons.
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        self.d_inputs[self.inputs <= 0] = 0


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


class AbstractLoss:
    def __init__(self):
        self.avg_loss = None
        self.output = None

    # Calculate the average loss of all the samples.
    def calculate_avg_loss(self, predictions, targets):
        self.forward(predictions, targets)
        self.avg_loss = np.mean(self.output)

    @abstractmethod
    def forward(self, predictions, targets):
        pass


class CategoricalCrossEntropy(AbstractLoss):
    def __init__(self):
        super().__init__()

    # Neuron1  2   3                    Targets
    # | 0.7, 0.1, 0.2   | Input1    | 1.0  0.0  0.0   |
    # | 0.1, 0.5, 0.4   | Input2    | 0.0, 1.0, 0.0   |
    # | 0.02, 0.9, 0.08 | Input3    | 0.0, 1.0, 0.0   |
    def forward(self, predictions, targets):
        clipped_predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        number_of_samples = len(clipped_predictions)

        if len(targets.shape) == 1:
            self.output = -1 * np.log(clipped_predictions[range(number_of_samples), targets])
        elif len(targets.shape) == 2:
            self.output = -1 * np.sum(targets * np.log(clipped_predictions), axis=1)
