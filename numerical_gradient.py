from functools import partial

import numpy as np

def compute(theta, J, epsilon=0.0001):
    """Accepts an array, and a function.
        Also optionally accepts an epsilon defining 
            neighborhood step size to check.

        Returns an array of partial derivatives.

        The partial derivatives are numerically computed
            by looking at the neighborhood around theta numerically.
    """
    assert theta.ndim == 1
    size = len(theta)
    grad = np.zeros(size)

    
    def offset(size, epsilon, i):
        """Accepts size int, epsilon real, and integer i.
            Returns new array of that size with zeros in almost all positions, 
                but epsilon in the ith position.
        """
        y = np.zeros(size)
        y[i] += epsilon
        return y

    o = partial(offset, size, epsilon)
    for i in xrange(size):
        e = o(i)
        grad[i] = (J(theta + e) - J(theta - e)) / (2 * epsilon)

    return grad

