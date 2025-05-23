import numpy as np

from nnfs.datasets import spiral_data
from main import DenseLayer, ReLuActivation, CompositeSoftmaxAndLoss, StochasticGradientDecrease, AdGrad

# [Definition] What is the optimizer?
# Answer: The optimizer adjust the weights and the biases to decrease the measure of the loss.

# [Definition] What is an epoch?
# Answer:   Each full pass through all of the training data is called an epoch. In neural networks it is done multiple times until
#           we reach a stop point.

# The aim is to create a neural network that can classify the data into the numeric equals of the colours, so into three classes.
# Each dot in the graph is a feature and its coordinates are the samples.
coordinates, classification = spiral_data(samples=100, classes=3)

def version_one():
    # Neural MNetwork
    v1_layer_1 = DenseLayer(2, 64)
    v1_activation_1 = ReLuActivation()

    v1_layer_2 = DenseLayer(64, 3)
    v1_composite_activation = CompositeSoftmaxAndLoss()

    v1_optimizer = StochasticGradientDecrease()

    # Training Loop
    for v1_epoch in range(10001):
        # Forward Pass
        v1_layer_1.forward(coordinates)
        v1_activation_1.forward(v1_layer_1.output)

        v1_layer_2.forward(v1_activation_1.output)
        v1_composite_activation.forward(v1_layer_2.output, classification)

        # if epoch % 100 == 0
        if not v1_epoch % 100:
            v1_acc = np.mean(np.argmax(v1_composite_activation.softmax.output, axis=1) == classification)
            v1_avg_loss = v1_composite_activation.loss.avg_loss
            print(f'epoch: {v1_epoch}, ' + f'acc: {v1_acc:.3f}, ' + f'loss: {v1_avg_loss:.3f}')

        # Backward Pass
        v1_composite_activation.backward(v1_composite_activation.softmax.output, classification)
        v1_layer_2.backward(v1_composite_activation.d_inputs)

        v1_activation_1.backward(v1_layer_2.d_inputs)
        v1_layer_1.backward(v1_activation_1.d_inputs)

        # Run Optimizer
        v1_optimizer.update_params(v1_layer_1)
        v1_optimizer.update_params(v1_layer_2)

        # After 10001 loop our optimization seems to stuck around loss: ~0.688 and accuracy: ~0.680.
        # This happens because the model get stuck in a local minimum so more iterations won't help.

if __name__ == '__main__':
    # Neural MNetwork
    layer_1 = DenseLayer(2, 64)
    activation_1 = ReLuActivation()

    layer_2 = DenseLayer(64, 3)
    composite_activation = CompositeSoftmaxAndLoss()

    # === SGD ===
    # 1e-2 = 0.01 -> Learning rate quickly became too small over the steps, model stuck in a local minimum.
    # optimizer = StochasticGradientDecrease(decay=1e-2)
    # 1e-3 = 0.001 -> Learning rate moderately became small over the steps, model gets closer to the global minimum then before.
    # optimizer = StochasticGradientDecrease(decay=1e-3)

    # === SGD (momentum) ===
    # 0.5 -> Achieved the best so far.
    # optimizer = StochasticGradientDecrease(decay=1e-3, momentum=0.5)
    # 0.9 -> Retaining more momentum, again, best so far, almost 90% accuracy.
    # epoch: 10000, acc: 0.947, loss: 0.127, learning rate: 0.091
    # optimizer = StochasticGradientDecrease(decay=1e-3, momentum=0.9)

    # === AdGrad (per-param) ===
    # epoch: 10000, acc: 0.917, loss: 0.226, learning rate: 0.500
    optimizer = AdGrad(decay=1e-4)

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
            print(f'epoch: {epoch}, ' + f'acc: {acc:.3f}, ' + f'loss: {avg_loss:.3f}, ' + f'learning rate: {optimizer.current_learning_rate:.3f}')

        # Backward Pass
        composite_activation.backward(composite_activation.softmax.output, classification)
        layer_2.backward(composite_activation.d_inputs)

        activation_1.backward(layer_2.d_inputs)
        layer_1.backward(activation_1.d_inputs)

        # Run Optimizer
        optimizer.pre_update_params()
        optimizer.update_params(layer_1)
        optimizer.update_params(layer_2)
        optimizer.post_update_params()
