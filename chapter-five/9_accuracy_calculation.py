# [Definition] What is accuracy?
# Answer:   How often the largest confidence is the correct class in terms of a fraction.

import numpy as np

# Probabilities of 3 samples
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])

# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])

# Get the index of the maximum value in each row
predictions = np.argmax(softmax_outputs, axis=1)

# If targets are one-hot encoded - convert them
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

# True evaluates to 1; False to 0, [0 1 1] -> 2/3 = 0.666666...
accuracy = np.mean(predictions == class_targets)

if __name__ == '__main__':
    print('Accuracy:', accuracy)
