import matplotlib.pyplot as plt
import nnfs
import numpy as np

from nnfs.datasets import spiral_data
from main import DenseLayer, SoftmaxActivation, CategoricalCrossEntropy, ReLuActivation

# Override numpy dot product method to ensure deterministic behaviour dues to random seeds for spiral data.
nnfs.init()

# The aim is to create a neural network that can classify the data into the numeric equals of the colours, so into three classes.
# Each dot in the graph is a feature and its coordinates are the samples.
coordinates, classification = spiral_data(samples=100, classes=3)

plt.scatter(coordinates[:, 0], coordinates[:, 1], c=classification, cmap='brg')
plt.show()

if __name__ == '__main__':
    layer_1 = DenseLayer(2, 3)
    layer_1_activation = ReLuActivation()

    layer_2 = DenseLayer(3, 3)
    layer_2_activation = SoftmaxActivation()

    loss_function = CategoricalCrossEntropy()

    layer_1.forward(coordinates)
    layer_1_activation.forward(layer_1.output)

    layer_2.forward(layer_1_activation.output)
    layer_2_activation.forward(layer_2.output)

    loss_function.calculate_avg_loss(layer_2_activation.output, classification)

    # At this point the neural network is based on random weights so the output is random.
    print("Layer 2 activation output: ", layer_2_activation.output[:5])
    # Loss is useful to optimize the model.
    print("Average loss: ", loss_function.avg_loss)
    # Accuracy of the model:
    accuracy = np.mean(np.argmax(layer_2_activation.output, axis=1) == classification)
    print("Accuracy: ", accuracy)
