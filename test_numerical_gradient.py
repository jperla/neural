import numpy as np

import numerical_gradient

def simple_quadratic(x):
    """Accepts a 2x1 vector x: (x1, x2).
        Returns the value of h(x) at x, 
            and its true gradient (partial derivatives w/ respect to x1, x2).
        h(x1, x2) = x1^2 + 3*x1*x2
    """
    value = x[0]**2 + (3 * x[0] * x[1])

    grad = np.zeros((2,));
    grad[0]  = (2 * x[0]) + (3 * x[1])
    grad[1]  = (3 * x[0])
    return value, grad

def norm(x):
    """Accepts n-d array. Calculates L2-norm.
        Returns real.
    """
    return np.sqrt(np.sum(x**2))

def diff_grad(g1, g2):
    """Accepts two gradients (arrays).
        Returns real.
        Computes normalized diff.
    """
    return norm(g1 - g2) / norm(g1 + g2)
    
def test_numerical_gradient():
    x = np.array([4, 10])
    epsilon = 0.0001
    less_than = 2.1453e-12
    ngrad = numerical_gradient.compute(x, 
                                       lambda i: simple_quadratic(i)[0], 
                                       epsilon)

    grad = simple_quadratic(x)[1]

    print ngrad, grad
    diff = diff_grad(ngrad, grad)
    print diff

    assert diff < less_than
    print 'Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n'

