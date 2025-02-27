import matplotlib.pyplot as plt
import numpy as np


# Linear Function = can be represented as a straight line.
def linear_function(x):
    return 2 * x


def non_linear_function(x):
    return 2 * x ** 2


if __name__ == '__main__':
    # [Definition] What is the slope (derivative - derivalt) of the function?
    # Answer: delta y (amennyit felfele megy) / delta x (amennyit elore megy)
    x = np.array(range(5))
    y = linear_function(x)
    print("Linear X: ", x)
    print("Linear Y: ", y)
    # delta y = y2 - y1 = 4 - 2
    # delta x = x2 - x1 = 2 - 1 (on the same x,y coordinate)
    # slope = y / x = 2 / 1 = 2
    plt.plot(x, y)
    plt.show()

    # How to calculate it?
    # Answer: We have to choose two infinitely close points (considering the restrictions of floating number calculations)
    y = non_linear_function(x)
    print("Non-Linear X:", x)
    print("Non-Linear Y:", y)

    delta = 0.0001 # A very small delta gives you a much accurate data for the slope at a point.
    x1 = 1
    x2 = x1 + delta

    y1 = non_linear_function(x1)
    y2 = non_linear_function(x2)

    slope = (y2 - y1) / (x2 - x1)
    print("Slope (x=1, y=2): ", slope)

    plt.plot(x, y)
    plt.show()

    # [Definition] What is numerical differentiation?
    # Answer: Calculating the slope of the tangent line using two infinitely close points.
