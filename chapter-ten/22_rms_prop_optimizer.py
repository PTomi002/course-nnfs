from math import sqrt

# [Definition]: RMSProp (Root Mean Square Propagation)
# Answer:   Similar as AdGrad with different cache calculation formula.

if __name__ == '__main__':
    param_gradient = 0.75

    # The cache carries so much momentum, smaller learning rate is enough.
    learning_rate = 0.001
    # rho = cache memory decay rate
    rho = 0.9
    cache = 0.
    eps = 1e-7

    # formula = retains a part of the cache +  updates it with the fraction of the squared gradient
    # so the "content" of the cache moves with update steps
    cache += rho * cache + ( 1 - rho) * param_gradient ** 2
    # Same as in AdGrad.
    param_update = learning_rate * param_gradient / sqrt(cache) + eps

