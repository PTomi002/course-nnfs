# [Definition] What are the analytic derivative rules?
# (1) Constant function is always zero: f(x) = 42 -> f'(x) = 0
# (2) Linear function:                  f(x) = x^n -> nx^(n-1), e.g.: f(x) = x -> 1x(1-1) = 1
# (3) Derivative of the sum:            f(x) + g(x) -> f'(x) + g'(x)
# (4) Derivative of the subtraction:    f(x) - g(x) -> f'(x) - g'(x)
# (5) etc...

if __name__ == '__main__':
    print("Function:    d/dx f(x)  = x^3 + 2x^2 - 5x + 7")
    print("Different notation (Leibniz):    df(x)/dx = x^3 + 2x^2 - 5x + 7")
    print("Derivative:  f'x(x) = 3x^2 + 4x - 5")
