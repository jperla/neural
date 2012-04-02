import numpy as np

import numerical_gradient

from testlib import diff_grad

def check_grad(sc, theta, threshold):
    cost, grad = sc(theta)
    ngrad = numerical_gradient.compute(theta,
                                       lambda x: sc(x)[0],
                                       epsilon=0.0001)
    print 'cost', cost
    print 'shapes:', ngrad.shape, grad.shape
    assert ngrad.shape == grad.shape

    diff = diff_grad(ngrad, grad)
    print 'diff:', diff

    assert diff < threshold
    return cost, grad, ngrad


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

