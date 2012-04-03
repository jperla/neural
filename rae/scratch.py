#!/usr/bin/env python
import numpy as np
import scipy.io

import neurolib

matlab_filename = 'data/rt-polaritydata/RTData_CV1.mat'
d = scipy.io.loadmat(matlab_filename)

np.sum(d['test_ind'])

def get_W(theta, has_Wcat, esize, cat_size, dictionary_length):
    """Accepts theta 1-d array of parameters,
        has_Wcat boolean of whether parameters have Wcat,bcat in it,
        esize integer size of embedded word vector size
        cat_size is number of dimensions in binary array 
            (1 for on binary categories, >1 for multinomial categories)
        dictionary_length integer number of words in dictionary
        Returns 10 arrays of all parameters for RAEs.
    """
    Wcat = []
    bcat = []
    vsize = esize * 2;
    wsize = esize*(vsize/2)

    W1, W2, W3, W4, b1, b2, b3, rest = np.split(theta, np.cumsum([wsize]*4 + [esize]*3))

    if has_Wcat:
        Wcat, bcat, We = np.split(rest, np.cumsum([cat_size*esize, cat_size]))
        Wcat.reshape(cat_size, esize)
        assert bcat.shape == cat_size
    else:
        We = rest

    W1.reshape(esize, vsize/2)
    W2.reshape(esize, vsize/2)
    W3.reshape(vsize/2, esize)
    W4.reshape(vsize/2, esize)
    We.reshape(esize, dictionary_length)
    
    assert b1.shape == b2.shape == b3.shape == (esize,)
                         
    return (W1, W2, W3, W4, b1, b2, b3, Wcat, bcat, We)


def initialize_params(esize, vsize, cat_size, dictionary_length):
    """
        Accepts esize integer of embedded representation size.
            vsize integer of input size into autoencoder,
                (typically a multiple of esize).
            cat_size integer of dimensionality of multinomial categories
            dictionary_length integer of number of words in vocab.
        Returns one flat array of parameters initialized randomly.

        Initialize parameters randomly based on layer sizes.
    """
    #We'll choose weights uniformly from the interval [-r, r]
    r  = np.sqrt(6) / np.sqrt(esize + vsize + 1)

    W1 = np.random.rand(esize, vsize) * 2 * r - r;
    W2 = np.random.rand(esize, vsize) * 2 * r - r;
    W3 = np.random.rand(vsize, esize) * 2 * r - r;
    W4 = np.random.rand(vsize, esize) * 2 * r - r;

    We = 1e-3 * (np.random.rand(esize, dictionary_length) * 2 * r - r)

    Wcat = np.random.rand(cat_size, esize) * 2 * r - r;

    b1 = np.zeros(esize, 1);
    b2 = np.zeros(vsize, 1);
    b3 = np.zeros(vsize, 1);
    bcat = np.zeros(cat_size, 1);

    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all
    # your parameters into a vector, which can then be used with minFunc.
    return neurolib.flatten_params(W1, W2, W3, W4, b1, b2, b3, Wcat, bcat, We)


def confusion_matrix(a, b):
    """Accepts two 1-d arrays (N,)-shape with integer class labels.
        Arrays must be of same size, and one predicts 
            the other class labels.
        Returns a 2-d array of NxN size.
    """
    N = a.shape[0]
    assert (N,) == a.shape == b.shape

    cm = np.zeros((N, N))
    from itertools import izip
    #TODO: jperla: might be faster with sparse matrices
    for i, j in izip(a, b):
        cm[i,j] += 1
    return cm

def  getAccuracy(predicted, gold):
    """Accepts two 2-d arrays of predicted and gold-truth class label
        integers from multinomial distribution.
        Returns 4-tuple of reals of (precision, recall, accuracy, F1 score)
    """
    cm = confusion_matrix(gold, predicted)
    n = len(gold)

    rowsums = np.max(np.vstack([np.sum(cm, axis=0), np.repeat(1e-5, n)]),
                     axis=0)
    colsums = np.max(np.vstack([np.sum(cm, axis=1), np.repeat(1e-5, n)]),
                     axis=0)

    precision = np.mean(np.diag(cm) / colsums)
    recall = np.mean(np.diag(cm) / rowsums)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, accuracy, f1

