import numpy as np

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
    
