# Manual representation of 1 neuron with 3 inputs in one shot.
input = [1.0, -2.0, 3.0]
weights = [- 3.0, -1.0, 2.0]
bias = 1.0

# Forward pass
o1 = input[0] * weights[0]
o2 = input[1] * weights[1]
o3 = input[2] * weights[2]

# Dot product: neuron output = sum(inputs x weights + bias)
sum = o1 + o2 + o3 + bias

# Activation function ReLU
z = max(sum, 0)

# Visualisation graph
#      (inputs x weights)
# 1.0  -----|
#         -3.0 -------
# -3.0 -----|        |
#                    |
# -2.0 -----|        V (sum)    (ReLU)
#         2.0 -----> 6.0 -----> 6.0 ---->
# -1.0 -----|        ^  ^
#                    |  |
# 3.0 -----|         |  |
#         6.0 --------  |
# 2.0 -----|            |
#   (bias)              |
# 1.0 -------------------

# Mathematical understanding
# formula =  ReLU(SUM(inputs x weights) + bias)
# formula =  ReLU(SUM(multiply(i[0], w[0]), multiply(i[1], w[1]), multiply(i[2], w[2]), bias))

# Backpropagation
d_value = 1.0  # assume that during backpropagation we got this value to ReLU from the next layer.

# Derivative of ReLU with respect to z and applied chain rule: x > 0 -> d/dx f(x) = x, f'(x) = 1
d_relu = d_value * (1. if z > 0 else 0.)

# Partial derivative of sum: e/ex f(x,y) = x + y, f'(x, y) = f'(x) = 1
d_sum_0 = 1
d_relu_sum_0 = d_relu * d_sum_0

d_sum_1 = 1
d_relu_sum_1 = d_relu * d_sum_1

d_sum_2 = 1
d_relu_sum_2 = d_relu * d_sum_2

d_sum_bias = 1
d_relu_sum_bias = d_relu * d_sum_bias

# Partial derivative of multiplication: f(x,y) = x * y -> f'x(x,y) = 1 * y
d_mu_we_0 = weights[0]
d_mu_we_0_d_sum_0 = d_relu_sum_0 * d_mu_we_0

d_mu_in_0 = input[0]
d_mu_in_0_d_sum_0 = d_relu_sum_0 * d_mu_in_0

d_mu_we_1 = weights[1]
d_mu_we_1_d_sum_1 = d_relu_sum_1 * d_mu_we_1

d_mu_in_1 = input[1]
d_mu_in_1_d_sum_1 = d_relu_sum_1 * d_mu_in_1

d_mu_we_2 = weights[2]
d_mu_we_2_d_sum_2 = d_relu_sum_2 * d_mu_we_2

d_mu_in_2 = input[2]
d_mu_in_2_d_sum_2 = d_relu_sum_2 * d_mu_in_2

# So the first weights formula (d_mu_we_0_d_sum_0) is (simplified):
# This is how we calculate the impact of the input to the neuron on the whole function’s output.
# d_mu_we_0_d_sum_0 = d_relu_sum_0 * d_mu_we_0
# d_mu_we_0_d_sum_0 = d_relu_sum_0 * weights[0]
# d_mu_we_0_d_sum_0 = d_relu * d_sum_0 * weights[0]
# d_mu_we_0_d_sum_0 = d_relu * d_sum_0 * weights[0]
# d_mu_we_0_d_sum_0 = d_relu * weights[0]
# d_mu_we_0_d_sum_0 = d_value * (1. if z > 0 else 0.) * weights[0]

# Gradient vectors
d_input = [d_mu_in_0_d_sum_0, d_mu_in_1_d_sum_1, d_mu_in_2_d_sum_2]
d_weights = [d_mu_we_0_d_sum_0, d_mu_we_1_d_sum_1, d_mu_we_2_d_sum_2]
d_bias = d_relu_sum_bias

if __name__ == '__main__':
    print("Forward pass result: ", z)
    print("Initial:             ", d_value)
    print("ReLU:                ", d_relu)
    print("Sum:                 ", d_relu_sum_0, d_relu_sum_1, d_relu_sum_2, d_relu_sum_bias)
    print("Multiplications:     ", d_mu_we_0_d_sum_0, d_mu_in_0_d_sum_0, d_mu_we_1_d_sum_1, d_mu_in_1_d_sum_1, d_mu_we_2_d_sum_2, d_mu_in_2_d_sum_2)
    print("Gradient vectors:    ", d_input, d_weights, d_bias)

    print("Applying optimization with a negative fraction to the weights to decrease the neuron's output and run forward pass again")
    print("Weights and bias: ", weights, bias)
    # Optimize
    weights[0] += -0.001 * d_weights[0]
    weights[1] += -0.001 * d_weights[1]
    weights[2] += -0.001 * d_weights[2]
    bias += -0.001 * d_bias

    print("Changed weights and bias: ", weights, bias)

    # Next forward pass
    o1 = input[0] * weights[0]
    o2 = input[1] * weights[1]
    o3 = input[2] * weights[2]

    sum = o1 + o2 + o3 + bias
    z = max(sum, 0)

    print("Optimized result:    ", z)

