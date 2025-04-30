import matplotlib.pyplot as plt
import nnfs
import numpy as np

from nnfs.datasets import spiral_data
from main import DenseLayer, SoftmaxActivation, CategoricalCrossEntropy, ReLuActivation, CompositeSoftmaxAndLoss

# Override numpy dot product method to ensure deterministic behaviour dues to random seeds for spiral data.
nnfs.init()

# The aim is to create a neural network that can classify the data into the numeric equals of the colours, so into three classes.
# Each dot in the graph is a feature and its coordinates are the samples.
coordinates, classification = spiral_data(samples=100, classes=3)

plt.scatter(coordinates[:, 0], coordinates[:, 1], c=classification, cmap='brg')
plt.show()


def first_version_code():
    # Neural Network
    layer_1 = DenseLayer(2, 3)
    layer_1_activation = ReLuActivation()

    layer_2 = DenseLayer(3, 3)
    layer_2_activation = SoftmaxActivation()

    loss_function = CategoricalCrossEntropy()

    # Forward Pass
    layer_1.forward(coordinates)
    layer_1_activation.forward(layer_1.output)

    layer_2.forward(layer_1_activation.output)
    layer_2_activation.forward(layer_2.output)

    loss_function.calculate_avg_loss(layer_2_activation.output, classification)

    # Print Results
    print("First 5 values: ", layer_2_activation.output[:5])
    print("Average loss: ", loss_function.avg_loss)
    print("Accuracy: ", np.mean(np.argmax(layer_2_activation.output, axis=1) == classification))

    # Backward Pass
    loss_function.backward(layer_2_activation.output, classification)
    layer_2_activation.backward(loss_function.d_inputs)
    layer_2.backward(layer_2_activation.d_inputs)
    layer_1_activation.backward(layer_2_activation.d_inputs)
    layer_1.backward(layer_1_activation.d_inputs)

    # Print Gradients
    print("Layer One Weights:", layer_1.d_weights)
    print("Layer One Biases:", layer_1.d_biases)
    print("Layer Two Weights:", layer_2.d_weights)
    print("Layer Two Biases:", layer_2.d_biases)

if __name__ == '__main__':
    # Neural Network
    l_1 = DenseLayer(2, 3)
    l_1_activation = ReLuActivation()

    l_2 = DenseLayer(3, 3)
    l_2_composite = CompositeSoftmaxAndLoss()

    # Forward Pass
    l_1.forward(coordinates)
    l_1_activation.forward(l_1.output)

    l_2.forward(l_1_activation.output)
    l_2_composite.forward(l_2.output, classification)

    # Print Results
    print("First 5 values: ", l_2_composite.softmax.output[:5])
    print("Average loss: ", l_2_composite.loss.avg_loss)
    print("Accuracy: ", np.mean(np.argmax(l_2_composite.softmax.output, axis=1) == classification))

    # Backward Pass
    l_2_composite.backward(l_2_composite.softmax.output, classification)
    l_2.backward(l_2_composite.d_inputs)

    l_1_activation.backward(l_2.d_inputs)
    l_1.backward(l_1_activation.d_inputs)

    # Print Gradients
    print("Layer One Weights:", l_1.d_weights)
    print("Layer One Biases:", l_1.d_biases)
    print("Layer Two Weights:", l_2.d_weights)
    print("Layer Two Biases:", l_2.d_biases)

    first_version_code()
