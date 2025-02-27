# [Definition] What is the gradient?
# Answer:  Is a vector composed of all the partial derivatives of a function.
# f(x,y,z) = 3(x^3)z - y^2 + 5z + 2yz all partial derivation:
#       |   z9x^2           | x
#       |   -2y + 2z        | y
#       |   3x^3 + 5 + 2y   | z

# [Definition] What is the chain rule?
# Answer:   To improve loss, we need to learn how each weight and bias impacts the loss at the end.
#           This rule says that the derivative of a function chain is a product of derivatives of all the functions in this chain.
#           We repeat this all the way down to the parameter in question!
#           d/dx f(g(x))         ->  f'x(g(x)) * g'x(x)
#           e/ex f(g(y,h(x,z)))  ->  f'x(g(y,h(x,z))) * g'x(y,h(x,z)) * h'x(x,z)

if __name__ == '__main__':
    print("Function:    h(x) = 3(2x^2)^5")
    print("y(x)   = 2x^2")
    print("h(x)   = 3(y)^5")
    print("h'x(x) = 15(y)^4 * y'")
    print("h'x(x) = 15(2x^2)^4 * 4x")

    print("Function:    h(x) = ln(sin(x))")
    print("y(x)   = sin(x)")
    print("h(x)   = ln(y)")
    print("h'x(x) = 1/y * y'")
    print("h'x(x) = 1/sin(x) * cos(x)")


