import matplotlib.pyplot as plt
import nnfs
import numpy as np

from nnfs.datasets import spiral_data
from main import DenseLayer, SoftmaxActivation, CategoricalCrossEntropy, ReLuActivation

nnfs.init()

coordinates, classification = spiral_data(samples=100, classes=3)

plt.scatter(coordinates[:, 0], coordinates[:, 1], c=classification, cmap='brg')
plt.show()

if __name__ == '__main__':
    layer_1 = DenseLayer(2, 3)
    layer_1_activation = ReLuActivation()

    layer_2 = DenseLayer(3, 3)
    layer_2_activation = SoftmaxActivation()

    loss_function = CategoricalCrossEntropy()

    lowest_loss = 999999
    best_layer_1_weights = layer_1.weights.copy()
    best_layer_1_biases = layer_1.biases.copy()
    best_layer_2_weights = layer_2.weights.copy()
    best_layer_2_biases = layer_2.biases.copy()

    for iteration in range(1000000):
        layer_1.weights += 0.05 * np.random.randn(2, 3)
        layer_1.biases += 0.05 * np.random.randn(1, 3)
        layer_2.weights += 0.05 * np.random.randn(3, 3)
        layer_2.biases += 0.05 * np.random.randn(1, 3)

        layer_1.forward(coordinates)
        layer_1_activation.forward(layer_1.output)

        layer_2.forward(layer_1_activation.output)
        layer_2_activation.forward(layer_2.output)

        loss_function.calculate_avg_loss(layer_2_activation.output, classification)

        if loss_function.avg_loss < lowest_loss:
            best_layer_1_weights = layer_1.weights.copy()
            best_layer_1_biases = layer_1.biases.copy()
            best_layer_2_weights = layer_2.weights.copy()
            best_layer_2_biases = layer_2.biases.copy()
            lowest_loss = loss_function.avg_loss
            print("Avg loss in iteration: ", iteration, " is", loss_function.avg_loss)
            print("Accuracy: ", np.mean(np.argmax(layer_2_activation.output, axis=1) == classification))
        else:
            layer_1.weights = best_layer_1_weights
            layer_1.biases = best_layer_1_biases
            layer_2.weights = best_layer_2_weights
            layer_2.biases = best_layer_2_biases

# We have to find a better way to optimize as "local loss" is happening, and it does not correlate with the complexity of the dataset.
# The "impact" of the input weights and biases flows through the whole Neural Network.
