# [Definition]: Learning Rate
# Answer:   We won't always apply negative gradient in the optimizers as-is, as the direction of the function steepest descent will
#               continuously changing. We want small steps (calculate gradient -> small parameter updates) and repeating it in a  loop.
#           Too small steps cause learning stagnation (model stuck in the local minimum).
#           Too big steps will cause gradient explosion.
#           Finding the global minimum with an n-dimensional (weights and biases) model is done by gradient descent algorithm.
# [Definition]: Momentum (Direction of the previous updates)
# Answer:   Momentum creates a rolling average of gradients over some number of updates and uses this
#               average with the unique gradient at each step.
# [Definition]: Learning Rate Decay
# Answer:   The idea of a learning rate decay is to start with a large learning rate, say 1.0 in our case, and
#               then decrease it during training.


if __name__ == '__main__':
    # [Definition]: Stochastic Gradient Descent
    # Basic optimization with fairly good results, but it just follows the opposite of the gradient without any extra logic.
    # It can stuck in a "closer to the global minimum" local minimum:
    starting_learning_rate = 1.
    learning_rate_decay = 0.1

    for step in range(100):
        # The further the training the bigger the value:    learning_rate_decay * step
        # The further the training the lower the value:     1. / denominator
        # But we don't want the learning rate be something huge because of: 1 /  a really small value
        #   -> Thus adding 1 + denominator.
        learning_rate = starting_learning_rate * (1. / (1 + learning_rate_decay * step))
        print(f"Learning Rate: {learning_rate}")

    # [Definition]: Stochastic Gradient Descent with Momentum
    #  Momentum uses the previous update’s direction to influence the next update’s direction, minimizing the chances of bouncing around and getting stuck.

