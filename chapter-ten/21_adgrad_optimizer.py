from math import sqrt

# [Definition]: Adaptive Gradient (AdGrad)
# Answer:   Works with a per-parameter learning rate instead of shared (decay, momentum) learning rate applied to all weights and biases.
#           Per-parameter updates -> Some parameters can change significantly during training while others not -> It auto updates learning rates.
#           Learning rate for parameters with smaller gradients -> Decrease slowly.
#           Learning rate for parameters with bigger gradients -> Decrease quicker.
#           No as good as SGD with decay + momentum.

if __name__ == '__main__':
    param_gradient = 0.75

    learning_rate = 1.
    cache = 0.
    eps = 1e-7

    # square = Measure the magnitude (how big these updates are), not the direction (positive or negative).
    cache += param_gradient ** 2
    # numerator 1st part = Basic SGD
    # eps = hyperparameter (pre-training setting) to prevent division by zero
    # squared root = scaling back the gradient updates from squares, do not want the denominator grow too fast
    param_update = learning_rate * param_gradient / sqrt(cache) + eps

