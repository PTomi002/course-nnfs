import matplotlib.pyplot as plt
import numpy as np

# See how weight and bias affects the slope of the linear function.
x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
w = 1.0
b = 0.0
y = (x * w) + b

plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5)
plt.plot(x, y)
plt.show()

# [Definition] What is a feature?
# Answer:   Features are the input variables that the model uses to make predictions or classifications.
#           The individual values of the feature set is a sample to the network.
#           A group of features makes up a feature set (represented as arrays/vectors).

inputs = [1.0, 2.0, 3.0, 2.5]
# our network will have weights initialized randomly, and biases set as zero to start
# use 3 neurons now, later we see how to determine their number
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
biases = [2, 3, 0.5]

# [Definition] What is a fully connected neural network?
# Answer: Every neuron in the current layer has connections to every neuron from the previous layer
outputs = [
    # Neuron 1
    inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + biases[0],
    # Neuron 2
    inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + biases[1],
    # Neuron 3
    inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + biases[2]
]

# matrix array (2-D array)
weights = [
    weights1,
    weights2,
    weights3
]


# make the calculation scalable
def calculate_outputs(inputs, weights, biases):
    layer_outputs = []
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for input, neuron_weight in zip(inputs, neuron_weights):
            neuron_output += input * neuron_weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)
    return layer_outputs


# [Definition] What is a tensor?
# Answer: A tensor is an object that can be represented as an array.

# [Definition] What is dot product?
# Answer: Sum of products of consecutive vector elements (what we did before ,but manually)
input = [1, 2, 3]
weight = [2, 3, 4]


def dot_product(arrayOne, arrayTwo):
    result = 0;
    for x, y in zip(arrayOne, arrayTwo):
        result += x * y
    return result


if __name__ == '__main__':
    print("Simple: ", outputs)
    print("Scalable: ", calculate_outputs(inputs, weights, biases))
    print("Dot Product: ", dot_product(input, weight))
