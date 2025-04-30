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
        self.d_inputs = np.dot(d_values, self.weights.T)
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
        self.d_inputs = None
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

    def backward(self, d_values):
        self.d_inputs = np.empty_like(d_values)
        for index, (output_row, d_values_row) in enumerate(zip(self.output, d_values)):
            # This reshapes the array into a 2D column vector.
            # The -1 means that the number of rows is inferred from the length of the array and the specified number of columns (which is 1 in this case).
            output_row = output_row.reshape(-1, 1)
            # Jacobian Matrix is an array of partial derivatives in all of the combinations.
            jacobian_matrix = np.diagflat(output_row) - np.dot(output_row, output_row.T)
            # The input (i1, i2, etc...) influences all the outputs, thus also influencing the partial derivative for each of them, so we have to sum them up.
            # Jacobian Matrix:     Gradients from next layer: (result as a row vector)
            #  | i1*i1   i1*i2   i1*i3 |  | g1 | = g1 * i1*i1 + g1 * i1*i2 + g1 * i1*i3 = r1
            #  | i2*i1   i2*i3   i2*i3 |  | g2 | = g2 * i2*i1 + g2 * i2*i2 + g2 * i2*i3 = r2
            #  | i3*i1   i3*i3   i3*i3 |  | g3 | = g3 * i3*i1 + g3 * i3*i2 + g3 * i3*i3 = r3
            # Results in a batch:
            # | r1,1 r1,2 r1,3 | r1,1 = Aggregated partial derivative for the influence feature 1. r1,2 = Aggregated partial derivative for the influence feature 2. Whole row = Aggregated influence of sample 1.
            # | r2,1 r2,2 r2,3 | Sample 2 result
            # | r3,1 r3,2 r3,3 | Sample 3 result
            self.d_inputs[index] = np.dot(jacobian_matrix, d_values_row)


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
        self.d_inputs = None

    # Neuron1  2   3                    Targets (one-hot vectors)
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

    def backward(self, d_values, y_true):
        number_of_features = len(d_values[0])
        # e.g.: arr = np.array([1, 2, 3, 4])   -> shape = (4,),   length is 1
        # e.g.: arr = np.array([[1, 2, 3, 4]]) -> shape = (2, 4), length is 2
        if len(y_true.shape) == 1:
            # If y_true is a row vector we convert it into one-hot vector matrix.
            y_true = np.eye(number_of_features)[y_true]

        self.d_inputs = -y_true / d_values

        size_of_batch = len(d_values)
        #  The optimizer will perform a sum operation so a sum divided by their count will bt the mean.
        # We dont want to adjust the learning rate based on the size of the samples, so we normalize it.
        self.d_inputs = self.d_inputs / size_of_batch


class CompositeSoftmaxAndLoss:
    def __init__(self):
        self.d_inputs = None
        self.softmax = SoftmaxActivation()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, targets):
        self.softmax.forward(inputs)
        self.loss.calculate_avg_loss(self.softmax.output, targets)

    def backward(self, d_values, y_true):
        # Turn one-hot vectors into one row vector.
        if len(y_true.shape) == 2:
            # y_true = [ 0 , 2 , 1 ]
            y_true = np.argmax(y_true, axis=1)

        self.d_inputs = d_values.copy()
        # range(len(d_values)) it will index all the rows, one-by-one as generates a sequence/range
        # y_true will index the columns
        size_of_batch = len(d_values)
        self.d_inputs[range(size_of_batch), y_true] -= 1
        #  The optimizer will perform a sum operation so a sum divided by their count will bt the mean.
        # We dont want to adjust the learning rate based on the size of the samples, so we normalize it.
        self.d_inputs = self.d_inputs / size_of_batch
