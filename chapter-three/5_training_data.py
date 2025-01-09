import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

# Override numpy dot product method to ensure deterministic behaviour dues to random seeds for spiral data.
nnfs.init()

# The aim is to create a neural network that can classify the data into the numeric equals of the colours, so into three classes.
# Each dot in the graph is a feature and its coordinates are the samples.
coordinates, classification = spiral_data(samples=100, classes=3)

plt.scatter(coordinates[:, 0], coordinates[:, 1], c=classification, cmap='brg')
plt.show()


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # We’re initializing weights to be (inputs, neurons) instead of transposing every time we perform a forward pass.
        # Multiply by 0.01 to make them smaller.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # The most common initialization for biases is 0.
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


if __name__ == '__main__':
    layer1 = DenseLayer(2, 3)
    layer1.forward(coordinates)
    print(layer1.output[:5])
