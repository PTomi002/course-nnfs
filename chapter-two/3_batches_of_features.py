import numpy as np

# Fitting the training data means: perform a step of training process.
# [Definition] Why do it in batches?
# Answer: (1) It is faster to do it parallel. (2) Gives you a higher chance of making more meaningful changes to weights and biases.

# Example batch of samples.
# 3x4 matrix
inputs = [
    [1, 2, 3, 2.5],
    [2, 5, - 1, 2],
    [- 1.5, 2.7, 3.3, - 0.8]
]

# [Definition] What is row vector?
# Answer: A row vector is a matrix with: 1xn dimension.
# [Definition] What is column vector?
# Answer: A column vector is a matrix with: nx1 dimension.
# [Definition] What is matrix transposition?
# Answer: Rows become columns and columns become rows.

row_vector = np.array([[1, 2, 3]])
column_vector = np.array([[2, 3, 4]]).T
matrix_product = np.dot(row_vector, column_vector)

# 3x4 matrix -transponate-> 4x3 matrix
weights = [
    [0.2, 0.8, - 0.5, 1],           # Neuron 1 Weights
    [0.5, - 0.91, 0.26, - 0.5],     # Neuron 2 Weights
    [- 0.26, - 0.27, 0.17, 0.87]    # Neuron 3 Weights
]
biases = [2.0, 3.0, 0.5]

#  3x4 4x3 matrix product
outputs = np.dot(inputs, np.array(weights).T) + biases
#                    Neuron1 Neuron2 Neuron3 --> Sample-related list of list instead of neuron-related list of list.
#  Input1 -> Output1 [[ 4.8    1.21   2.385]
#  Input2 -> Output2 [ 8.9   -1.81   0.2  ]
#  Input3 -> Output3 [ 1.41   1.051  0.026]]

if __name__ == '__main__':
    print("Row vector: ", row_vector)
    print("Column vector: ", column_vector)
    print("Matrix product: ", matrix_product)
    print("Layer output: ", outputs)
