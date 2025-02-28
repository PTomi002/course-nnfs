# [Definition] What is the partial derivative?
# Answer:   It measures how much impact a single input has on a function’s output.
#           We try to find the impact of the given input to the output while treating all of the
#               other inputs as constants.

# [Definition] What are the partial derivative rules?
# (1) Multiplication:   f(x,y) = x * y      -> f'x(x,y) = 1 * y // we derivate x, (c*f) -> c*f'
#                                           -> f'y(x,y) = x * 1 // we derivate y, constants move out of the derivation
# (2) Max function:     f(x,y) = max(x,y)   -> f'(x,y)  = 1(x > y) // 1 if the condition is met, otherwise 0
# Imagine it for the different intervals:
# x > 0 -> d/dx f(x) = x, f'(x) = 1
# x < 0 -> d/dx f(x) = 0, f'(x) = 0

if __name__ == '__main__':
    print("Function:                    f(x,y) = 3x^3 - y^2 + 5x")
    print("Partial derivative for x:    e/ex f(x,y) -> f'x(x,y) = 9x^2 + 5") # y treated as constant, because not affected by x
    print("Partial derivative for y:    e/ey f(x,y) -> f'y(x,y) = -2y") # x treated as constant, because not affected by y

    print("Function:                    f(x,y,z) = 3(x^3)z - y^2 + 5z + 2yz")
    print("Partial derivative for x:    e/ex f(x,y,z) -> f'x(x,y,z) = z9x^2") # (3z*(x^3)) -> 3z*(3x^2)
    print("Partial derivative for y:    e/ey f(x,y,z) -> f'y(x,y,z) = -2y + 2z")
    print("Partial derivative for z:    e/ez f(x,y,z) -> f'z(x,y,z) = 3x^3 + 5 + 2y")
