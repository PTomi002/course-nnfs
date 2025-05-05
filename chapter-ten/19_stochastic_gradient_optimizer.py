import numpy as np

from nnfs.datasets import spiral_data
from main import DenseLayer, ReLuActivation, CompositeSoftmaxAndLoss, StochasticGradientDecrease

# [Definition] What is the optimizer?
# Answer: The optimizer adjust the weights and the biases to decrease the measure of the loss.

# [Definition] What is an epoch?
# Answer:   Each full pass through all of the training data is called an epoch. In neural networks it is done multiple times until
#           we reach a stop point.

# The aim is to create a neural network that can classify the data into the numeric equals of the colours, so into three classes.
# Each dot in the graph is a feature and its coordinates are the samples.
coordinates, classification = spiral_data(samples=100, classes=3)

if __name__ == '__main__':
    # Neural MNetwork
    layer_1 = DenseLayer(2, 64)
    activation_1 = ReLuActivation()

    layer_2 = DenseLayer(64, 3)
    composite_activation = CompositeSoftmaxAndLoss()

    optimizer = StochasticGradientDecrease()

    # Training Loop
    for epoch in range(10001):
        # Forward Pass
        layer_1.forward(coordinates)
        activation_1.forward(layer_1.output)

        layer_2.forward(activation_1.output)
        composite_activation.forward(layer_2.output, classification)

        # if epoch % 100 == 0
        if not epoch % 100:
            acc = np.mean(np.argmax(composite_activation.softmax.output, axis=1) == classification)
            avg_loss = composite_activation.loss.avg_loss
            print(f'epoch: {epoch}, ' + f'acc: {acc:.3f}, ' + f'loss: {avg_loss:.3f}')

        # Backward Pass
        composite_activation.backward(composite_activation.softmax.output, classification)
        layer_2.backward(composite_activation.d_inputs)

        activation_1.backward(layer_2.d_inputs)
        layer_1.backward(activation_1.d_inputs)

        # Run Optimizer
        optimizer.update_params(layer_1)
        optimizer.update_params(layer_2)

        # After 10001 loop our optimization seems to stuck around loss: ~0.688 and accuracy: ~0.680.
        # This happens because the model get stuck in a local minimum so more iterations won't help.
