# [Definition] What is the gradient?
# Answer:  Is a vector composed of all the partial derivatives of a function with respect to each inputs and weights and bias.
# f(x,y,z) = 3(x^3)z - y^2 + 5z + 2yz all partial derivation:
#       |   z9x^2           | x
#       |   -2y + 2z        | y
#       |   3x^3 + 5 + 2y   | z

# [Definition] What is the chain rule?
# Answer:   To improve loss, we need to learn how each weight and bias impacts the loss at the end.
#           This rule says that the derivative of a function chain (composite function) is a product of derivatives of all the functions in this chain:
#           d/dx f(g(x))         ->  df(g(x))/dg(x) * dg(x)/dx                                          ->  f'x(g(x)) * g'x(x)
#
#           Partial differentiation with chain rule for X variable:
#           e/ex f(g(y,h(x,z)))  ->  df(g(y,h(x,z)))/dg(y,h(x,z)) * dg(y,h(x,z))/dh(x,z) * dh(x,z)/dx   -> f'x(g(y,h(x,z))) * g'x(y,h(x,z)) * h'x(x,z)
#
#           Explanation:
#               Imagine the dependency graph of the function f(x,y,z):    f
#                                                                         |         df/dg
#                                                                         V
#                                                                     --- g ---
#                                                                     |       |     dg/dh
#                                                                     V       V
#                                                                     y   -- h --
#                                                                         |     |   dh/dx
#                                                                         V     V
#                                                                         x     z
#
#               We want to move through all the branches which terminates at X.
#               So finally got:
#                ef   df   dg   dh
#                -- = -- * -- * --
#                ex   dg   dh   dx

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

    print("Function:    h(x) = (2x^3 + 5)^7")
    print("h'x(x) = 7(2x^3 + 5)^6 * 6x^2")

    print("Function:    h(x) = e^ln(x^2 + 1)")
    print("h(x) = de^ln(x^2+1)/dln(x^2+1) * dln(x^2+1)/dx^2+1 * dx^2+1/dx")
    print("h'x(x) = (x^2+1) * 1/x^2+1 * 2x")  # The simplification/derivative of e^ln(a) is a.
