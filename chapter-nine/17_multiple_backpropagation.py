import numpy as np

# Forward pass output
z = np.array([
    # Neuron 1 ... Neuron 3
    [1, 2, - 3],    # Output 1
    [2, - 7, - 1],  # Output 2
    [-1, 2, 5]      # Output 3
])

# Gradient vector from the next layer compared to the current layer.
# For 3 input vectors we have 3 gradient vectors.
d_values = np.array([
    #  Neuron 1... Neuron 3
    [1., 1., 1.],  # Gradient 1 / For Sample 1
    [2., 2., 2.],  # Gradient 2 / For Sample 2
    [3., 3., 3.]   # Gradient 3 / For Sample 3
])

weights = np.array([
# Weight1 , ...., Weight 4
    [0.2, 0.8, -0.5, 1],        # Neuron 1
    [0.5, -0.91, 0.26, -0.5],   # Neuron 2
    [-0.26, -0.27, 0.17, 0.87]  # Neuron 3
])

inputs = np.array([
# Feature 1 .... Feature 4
    [ 1 , 2 , 3 , 2.5 ],            # Sample 1
    [ 2. , 5. ,- 1. , 2 ],          # Sample 2
    [ - 1.5 , 2.7 , 3.3 ,- 0.8 ]    # Sample 3
])

#           Neuron 1 .. Neuron 3
# Weight 1    0.2         - 0.26
# ...
# Weight 4     1          0.87
weights_t = weights.T

# Aggregate the input's total influence on the neurons' outputs.
d_x0 = sum(weights_t[0] * d_values[0])  # d_values[0] means a single input sample
d_x1 = sum(weights_t[1] * d_values[0])
d_x2 = sum(weights_t[2] * d_values[0])
d_x3 = sum(weights_t[3] * d_values[0])

# This is the gradient of the neuron function with respect to inputs.
d_inputs = np.array([d_x0, d_x1, d_x2, d_x3])

# (1) Calculating the gradients with respect to inputs:
# The partial derivative with respect to the input equals the related weight.
# d_mu_we_0_d_sum_0 = d_value (g11) * (1. if z > 0 else 0.) * weights[0]
# d_mu_we_1_d_sum_1 = d_value (g11) * (1. if z > 0 else 0.) * weights[1]
# d_inputs_simplified = (inputok) X (feature-ok) gradiense
#                                                    Weight 1  ....  Weight 4
#                                         Neuron 1      w11             w41
#                                         Neuron 2      w12             w42
#                                         Neuron 3      w13             w43
#
#   Neuron 1  ......   Neuron 3                      Feature 1    ...    Feature 4
#        g11             g13     Gradient 1            g11              g14     Gradient 1
#        g21             g23     Gradient 2            g21              g24     Gradient 2
#        g31             g33     Gradient 3            g31              g34     Gradient 3
d_inputs_simplified = np.dot(d_values, weights)

# (2) Calculating the gradients with respect to weights:
# The partial derivative with respect to the weights equals the related input.
# d_mu_in_0_d_sum_0 = d_value (g11) * (1. if z > 0 else 0.) * inputs[0]
# d_mu_we_1_d_sum_1 = d_value (g11) * (1. if z > 0 else 0.) * inputs[1]
# d_weights_simplified = (feature-ok) X (neuronok) gradiense
#                                                       Neuron 1  ....  Neuron 3
#                                          Gradient 1     g11             g13
#                                          Gradient 2     g21             g23
#                                          Gradient 3     g31             g33
#
#   Sample 1     ....   Sample 3                       Neuron 1  ...  Neuron 3
#        i11             i31     Feature 1              g11              g31     Feature 1
#        i12             i32     Feature 2              g12              g32     Feature 2
#        i13             i33     Feature 3              g13              g33     Feature 3
#        i14             i34     Feature 4              g14              g34     Feature 4
# This is the gradient of the neuron function with respect to the weights.
d_weights_simplified = np.dot(inputs.T, d_values)

# (3) Calculating the bias:
# Since the bias affects all samples in the batch, we sum the gradients over all samples by neurons.
# The derivative of the bias is 1, so we have to only sum the gradients.
# d_relu_sum_bias = d_value * (1. if z > 0 else 0.)
d_bias_simplified = np.sum(d_values, axis=0, keepdims=True)

# (4) Calculating ReLU:
# d_relu = d_value * (1. if z > 0 else 0.)
d_values_for_relu = np.array([
    [ 1 , 2 , 3 ],
    [ 5 , 6 , 7 ],
    [ 9 , 10 , 11]
])
d_relu = d_values_for_relu.copy()
d_relu[z <= 0 ] = 0

if __name__ == '__main__':
    print(d_inputs_simplified)
    print(d_weights_simplified)
    print(d_bias_simplified)
    print(d_relu)
