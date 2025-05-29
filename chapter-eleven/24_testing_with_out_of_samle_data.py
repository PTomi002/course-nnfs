import numpy as np
import matplotlib.pyplot as plt

from nnfs.datasets import spiral_data

from main import DenseLayer, ReLuActivation, CompositeSoftmaxAndLoss, Adam

# [Definition] Overfitting
# Answer:   The model just memorize the data it has seen,it is good at recognising it, but bad at generalisation (prediction) with never seen inputs.
#           It is essential to have training and test data as separate sets.
#           We must isolate the test data well enough from the training data.

training_coordinates, training_classification = spiral_data(samples=100, classes=3)
test_coordinates, test_classification = spiral_data(samples=100, classes=3)

training_steps = np.arange(0, 10001)
loss_over_training_steps = []
acc_over_training_steps = []
learning_rate_over_training_steps = []

if __name__ == '__main__':
    # Neural MNetwork
    layer_1 = DenseLayer(2, 64)
    activation_1 = ReLuActivation()

    layer_2 = DenseLayer(64, 3)
    composite_activation = CompositeSoftmaxAndLoss()

    optimizer = Adam(learning_rate=0.05, decay=5e-7)  # 0.0000005

    # Training Loop
    for epoch in range(len(training_steps)):
        # Forward Pass
        layer_1.forward(training_coordinates)
        activation_1.forward(layer_1.output)

        layer_2.forward(activation_1.output)
        composite_activation.forward(layer_2.output, training_classification)

        acc = np.mean(np.argmax(composite_activation.softmax.output, axis=1) == training_classification)
        acc_over_training_steps.append(acc)

        avg_loss = composite_activation.loss.avg_loss
        loss_over_training_steps.append(avg_loss)

        learning_rate_over_training_steps.append(optimizer.current_learning_rate)

        if epoch == training_steps.size - 1:
            print(f'epoch: {epoch}, ' + f'acc: {acc:.3f}, ' + f'loss: {avg_loss:.3f}, ' + f'learning rate: {optimizer.current_learning_rate:.10f}')

        # Backward Pass
        composite_activation.backward(composite_activation.softmax.output, training_classification)
        layer_2.backward(composite_activation.d_inputs)

        activation_1.backward(layer_2.d_inputs)
        layer_1.backward(activation_1.d_inputs)

        # Run Optimizer
        optimizer.pre_update_params()
        optimizer.update_params(layer_1)
        optimizer.update_params(layer_2)
        optimizer.post_update_params()

    # Validating the model
    layer_1.forward(test_coordinates)
    activation_1.forward(layer_1.output)

    layer_2.forward(activation_1.output)
    composite_activation.forward(layer_2.output, test_classification)

    acc = np.mean(np.argmax(composite_activation.softmax.output, axis=1) == test_classification)
    avg_loss = composite_activation.loss.avg_loss

    print(f'acc: {acc:.3f}, ' + f'loss: {avg_loss:.3f}')

    # During training the model we reach:   epoch: 10000, acc: 0.953, loss: 0.123, learning rate: 0.0497512685
    # During prediction we reach:           acc: 0.767, loss: 1.611
    # The difference is so huge (good sign if: it is more than 10%), it must be an overfitting issue.
    # The goal is to have the testing loss identical to the training loss --> Similar loss means that the model is generalized, even with less accuracy.
    fig, axs = plt.subplots(2)

    axs[0].set_title('Training Steps')
    axs[0].plot(training_steps, np.array(loss_over_training_steps), color='green', label='loss')
    axs[0].plot(training_steps, np.array(acc_over_training_steps), color='blue', label='accuracy')
    axs[0].legend()

    axs[1].set_title('Training Steps')
    axs[1].plot(training_steps, np.array(learning_rate_over_training_steps), color='red', label='learning rate')
    axs[1].legend()

    plt.show()
    # Classic example of overfitting when the loss starts to fall, then starts rising again.

    # [Definition] How to find the hyperparameters?
    # Answer:   ???

