#!/usr/bin/env python
import numpy as np

epsilon = 0.000000001

def T(a):
    return a.reshape(len(a), 1)

def zero_mean(data):
    """Accepts columns of data.
        Zeros out the mean of the data.
        Each column will have zero mean.
    """
    return data - np.mean(data, axis=0)

def covariance(data):
    """Accepts columns of data.
        Calculates the covariance matrix.
        Returns new array.
    """
    return np.dot(data, data.T) / (data.shape[1])

def pca(data, k=None):
    """Accepts columns of data, and number of dimensions to reduce to.
        Assumes that each column has mean of zero.
        Performs PCA.
        Returns the eigenvectors and eigenvalues.
        Also returns new array of smaller dimensions.
            Whereas data was DxN array, 
            new array is kxN.
            (if K=None, then k=D)
    """
    assert k is None or 1 <= k <= data.shape[0]

    sigma = covariance(data)
    U, s, V = np.linalg.svd(sigma)

    if k is None:
        x_rot = np.dot(U.T, data)
    else:
        x_rot = np.dot((U[:,:k]).T, data)
    return U, s, x_rot

def pca_whiten(data, k=None, epsilon=0.00001):
    """Accepts columns of data.
        Assumes that each column has mean of zero.
        PCA whitens the data.
        Returns new array of same size.
    """
    U, s, x_rot = pca(data, k)
    return U, s, x_rot / T(np.sqrt(s) + epsilon)

def zca_whiten(data, k=None, epsilon=0.00001):
    """Accepts columns of data.
        Assumes that each column has mean of zero.
        PCA whitens the data.
        Returns new array of same size.
    """
    U, s, x_rot = pca(data, k)
    return U, s, np.dot(U, x_rot / T(np.sqrt(s) + epsilon))

