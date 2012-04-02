import numpy as np

def T(a):
    """Given 1-d array. Make it a column vector.
        Returns 2d array with Nx1 size.

        Useful when adding a column vector to every column in a matrix.
            (instead of bsxfun or repmat in Matlab)
    """
    return a.reshape(len(a), 1)

def sigmoid(x): 
    """Accepts real.
        Returns real.
        Sigmoid function: range is [0,1].
        Similar to tanh function.
    """
    return 1.0 / (1.0 + np.exp(-x))

def binary_KL_divergence(p1, p2):
    """Accepts two real numbers.
        Works if one of them is an numpy array (returns an array of reals).
        Returns the a real, KL divergence of two binomial distributions with 
            probabilities p1 and p2 respectively.
    """
    return (p1 * np.log(p1/p2)) + ((1 - p1) * np.log((1 - p1) / (1 - p2)))

