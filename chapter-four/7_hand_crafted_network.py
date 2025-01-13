import numpy as np
import math
import matplotlib.pyplot as plt

inputs = np.linspace(0, 1, 100)
sin = np.sin(2 * math.pi * inputs)

plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5)
plt.plot(inputs, sin)

# Handcrafted and Fitted Neural Network for sin wave, only the first section is done.
# Used ReLU activation to represent the area of effect for each neuron.
# Hidden Layer 1
l1_w = np.array([  # +,- -> mirroring, value -> slope of the overall function
    6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])
l1_b = np.array([  # value -> vertical movement
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# Hidden Layer 2
l2_w = np.array([
    -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])
l2_b = np.array([
    0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
])

# Output Layer
o_w = np.array([
    -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])


def relu(x):
    # return np.where(x > 0, x, 0)
    return np.where(x < 0, x, 0)


plot_output = []
for i in inputs:
    l1_out = []
    l2_out = []
    for w, b in zip(l1_w, l1_b):
        l1_out.append(i * w + b)
    for o, w, b in zip(l1_out, l2_w, l2_b):
        l2_out.append(o * w + b)
    sum_of_output_neuron = 0
    for o, w in zip(l2_out, o_w):
        sum_of_output_neuron += o * w
    plot_output.append(sum_of_output_neuron)

if __name__ == '__main__':
    plt.plot(inputs, relu(np.array(plot_output)))
    plt.show()
