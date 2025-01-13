import numpy
import numpy as np
import matplotlib.pyplot as plt

# [Definition] What is activation function?
# Answer: The purpose this activation function serves is to mimic a neuron “firing” or “not firing” based on input information.

# [Definition] Which activation function to choose from on the dense layers?
# Answer: Sigmoid vs ReLU??

# [Definition] What is Sigmoid?
# Answer:   See below. Only for historical purposes.

x = np.linspace(-15, 15, 50)
sigmoid_y = 1 / (1 + np.exp(-1 * x))

plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5)
plt.plot(x, sigmoid_y)
plt.show()

# [Definition] What is ReLU (Rectified Linear Units)?
# Answer:   See below.
#           Mostly used nowadays due to its speed and efficiency.

relu_y = np.maximum(0, x)

plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5)
plt.plot(x, relu_y)
plt.show()

# [Definition] Which activation function to choose from on the output layer?
# Answer: It depends on our aims, we want classification, so SoftMax is what we need.

# [Definition] What is Softmax?
# Answer:   Softmax activation on the output data can take in non-normalized, or uncalibrated, inputs and
#           produce a normalized distribution of probabilities for our classes. The distribution returned by the softmax
#           activation function represents confidence scores for each class and will add up to 1.

layer_outputs = [4.8, 1.21, 2.385]


def softmax(inputs):
    # Unnormalized probabilities.
    exp_values = np.exp(inputs)     # We need non-negative values for probabilities.
                                    # Adds stability as it does not change their classes but highlight the difference between them.
    return exp_values / np.sum(exp_values)

# Issues with softmax, the exponential function -> It can easily explode.
# Subtract the maximum number would rearrange the numbers from:
#     - the maximum will be zero
#     - anything else will be negative

if __name__ == '__main__':
    print("Softmax output: ", softmax(layer_outputs))
