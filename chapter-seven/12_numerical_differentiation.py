import matplotlib.pyplot as plt
import numpy as np


def non_linear_function(x):
    return 2 * x ** 2


def tangent_line(m, x, b):
    return m * x + b


if __name__ == '__main__':
    # Like the np.linspace(...) but this defines the step value instead of the number of steps within the interval.
    x = np.arange(0, 5, 0.001)
    y = non_linear_function(x)

    # Appr. derivative at x = 1, y =2
    delta = 0.0001
    x1 = 1
    x2 = x1 + delta
    y1 = non_linear_function(x1)
    y2 = non_linear_function(x2)
    slope = (y2 - y1) / (x2 - x1)
    print("Slope (x=1, y=2): ", slope)

    # Try to draw the tangent line (erinto vonal) on X1.
    # y = m*x + b -> y - mx = b
    b = y1 - slope * x1
    y_tangent_line = tangent_line(slope, x, b)

    plt.plot(x, y)
    plt.plot(x, y_tangent_line)
    plt.show()

    # We have to understand the impact input X because:
    # Answer:   It informs us about the impact that x has on this function at a particular point.
    #           It means that for a very small step in X how much my value will change.