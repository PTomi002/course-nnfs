# [Definition] What is Categorical cross-entropy?
# Answer:   Explicitly used to compare “ground-truth” probability (called "y" or "targets") and some predicted distribution by the neural network (called "y-hat").
#           Outputs higher loss for lower confidence.
import numpy as np
import math
import matplotlib.pyplot as plt
from main import CategoricalCrossEntropy

# Let's say the first value is the desired prediction in the 3 way classification.
softmax_output = np.array([0.7, 0.1, 0.2])

# The desired probability should look like this.
# This is called "one-hot" vector, because the other value are zero.
target_distribution = np.array([1.0, 0.0, 0.0])


def categorical_cross_entropy(target, output):
    return -1 * np.sum(target * np.log(output))


# We can make it more simple, by omitting the zero multiplications
# k is the index of the true probability from the targets
def simplified_categorical_cross_entropy(output, k):
    return -1 * np.log(output[k])


def batch_categorical_cross_entropy(outputs, targets):
    if len(targets.shape) == 1:
        return -1 * np.log(outputs[range(len(outputs)), targets])
    elif len(targets.shape) == 2:
        return -1 * np.sum(targets * np.log(outputs), axis=1)
    return None


# Natural Logarithm: a^x = b -> log a(b) = x | e^x = 5.2 -> ln(5.2) = x
# Why logarithm? Later....
x = np.linspace(-100, 100, 500)
y = np.log(x)

plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5)
plt.plot(x, y)
plt.show()

# NumPy advanced indexing with batch of inputs
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])

# See how the method works:
class_targets_batch = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 1, 0]])

# Divided by zero issues for the softmax_outputs:
# (1) If zero is used in the log -> Adding a very small value to it: 1e-7
# (2) If one is used in the log -> Use slightly less value then one: 1-1e-7

if __name__ == '__main__':
    print("Categorical cross-entropy: ", categorical_cross_entropy(target_distribution, softmax_output))
    print("Simplified Categorical cross-entropy: ", simplified_categorical_cross_entropy(softmax_output, 0))

    log = np.log(5.2)
    print("ln(5.2):", log)
    print("e^1.6486586255873816: ", np.pow(math.e, log))
    print("Natural logarithm for zero: ", -1 * np.log(0.0))
    print("Adding a very small value instead of zero confidence result in loss ", -1 * np.log(1e-7))
    print("Natural logarithm for one: ", -1 * np.log(1.0))
    print("Adding a slightly less than one value instead of one confidence results in loss: ", -1 * np.log(1 - 1e-7))
    print("Clipping the small value in the array: ", np.clip(class_targets_batch, 1e-7, 1-1e-7))

    print("Advanced indexing (select rows only): ", softmax_outputs[class_targets])

    print("Loss of batch of inputs: ", batch_categorical_cross_entropy(softmax_outputs, class_targets))
    print("Loss of batch of inputs: ", batch_categorical_cross_entropy(softmax_outputs, class_targets_batch))
    print("Average loss of batch of inputs: ", np.mean(batch_categorical_cross_entropy(softmax_outputs, class_targets)))

    loss = CategoricalCrossEntropy()
    loss.calculate_avg_loss(softmax_outputs, class_targets)
    print("Loss output: ", loss.output)
    print("Loss avg: ", loss.avg_loss)
