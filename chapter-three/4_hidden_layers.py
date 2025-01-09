import numpy as np

# [Definition] What is a hidden layer?
# Answer: Layers between the input and the output layer.

inputs = [
    [1, 2, 3, 2.5],
    [2, 5, - 1, 2],
    [- 1.5, 2.7, 3.3, - 0.8]
]

hidden_layer_1_weights = [
    [0.2, 0.8, - 0.5, 1],
    [0.5, - 0.91, 0.26, - 0.5],
    [- 0.26, - 0.27, 0.17, 0.87]
]
hidden_layer_1_biases = [2.0, 3.0, 0.5]

hidden_layer_2_weights = [
    [0.1, - 0.14, 0.5],
    [- 0.5, 0.12, - 0.33],
    [- 0.44, 0.73, - 0.13]
]
hidden_layer_2_biases = [- 1, 2, - 0.5]

hidden_layer_1_output = np.dot(inputs, np.array(hidden_layer_1_weights).T) + hidden_layer_1_biases
hidden_layer_2_output = np.dot(hidden_layer_1_output, np.array(hidden_layer_2_weights).T) + hidden_layer_2_biases

if __name__ == '__main__':
    print("Hidden layer 1 output: ", hidden_layer_1_output)
    print("Hidden layer 2 output: ", hidden_layer_2_output)
