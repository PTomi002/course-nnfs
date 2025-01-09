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

if __name__ == '__main__':
    print(outputs)
