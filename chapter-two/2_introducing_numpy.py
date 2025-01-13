import numpy as np

# We call this a sample or a feature set or an observation.
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [
    [0.2, 0.8, - 0.5, 1],
    [0.5, - 0.91, 0.26, - 0.5],
    [- 0.26, - 0.27, 0.17, 0.87]
]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(weights, inputs) + biases

# Axis during summarization.
layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, - 1.81, 0.2],
                          [1.41, 1.051, 0.026]])

if __name__ == '__main__':
    print(outputs)
    print("Sum without axis: ", np.sum(layer_outputs))
    print("Sum with axis=0 (sum of columns): ", np.sum(layer_outputs, axis=0))
    print("Sum with axis=1 (sum of rows): ", np.sum(layer_outputs, axis=1))
    print("Sum with axis=1 (sum of rows) and keep dimensions: ", np.sum(layer_outputs, axis=1, keepdims=True))

    print("Max without axis: ", np.max(layer_outputs))
    print("Max with axis=1 (max of rows): ", np.max(layer_outputs, axis=1))
    print("Max with axis=1 (max of rows) and keep dimensions: ", np.max(layer_outputs, axis=1, keepdims=True))

    print("Input - maximum: ", layer_outputs - np.max(layer_outputs, axis=1, keepdims=True))
