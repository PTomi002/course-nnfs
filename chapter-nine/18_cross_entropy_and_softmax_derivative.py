import numpy as np
import nnfs

from main import SoftmaxActivation, CategoricalCrossEntropy, CompositeSoftmaxAndLoss

nnfs.init()

softmax_output = [0.7, 0.1, 0.2]
softmax_output = np.array(softmax_output).reshape(-1, 1)
kronecker_delta = np.eye(softmax_output.shape[0])

left_side_of_the_equation = softmax_output * kronecker_delta
right_side_of_the_equation = np.dot(softmax_output, softmax_output.T)


def check_different_solutions():
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])
    class_targets = np.array([0, 1, 1])

    # Composite
    composite = CompositeSoftmaxAndLoss()
    composite.backward(softmax_outputs, class_targets)

    # Separate
    softmax = SoftmaxActivation()
    softmax.output = softmax_outputs

    loss = CategoricalCrossEntropy()
    loss.backward(softmax_outputs, class_targets)
    softmax.backward(loss.d_inputs)

    # Check
    print('Gradients: combined loss and activation:')
    print(composite.d_inputs)
    print('Gradients: separate loss and activation:')
    print(softmax.d_inputs)

if __name__ == '__main__':
    print("See my notebook for the solutions.")
    print("softmax output:\n", softmax_output)
    print("softmax shape: ", softmax_output.shape)
    print("kronecker delta:\n", kronecker_delta)
    print("left side:\n", left_side_of_the_equation)
    print("diagflat\n", np.diagflat(softmax_output))

    # In Softmax each input influences each output.
    print("reshaped softmax transponate\n", softmax_output.T)
    print("right side (jacobi matrix):\n", right_side_of_the_equation)
    print("solution:\n", left_side_of_the_equation - right_side_of_the_equation)
    print("solution shape:\n", (left_side_of_the_equation - right_side_of_the_equation).shape)

    test = np.array([
        [1, 2, 1],
        [2, 2, 2],
        [3, 3, 3]
    ])

    print("test\n", test)
    print("result\n", np.dot(test, np.array([1, 2, 3])))

    check_different_solutions()