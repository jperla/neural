import numpy as np
import scipy.io

matlab_filename = 'data/rt-polaritydata/RTData_CV1.mat'
d = scipy.io.loadmat(matlab_filename)

np.sum(d['test_ind'])

def get_W(theta, has_Wcat, esize, cat_size, dictionary_length)
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
    r  = sqrt(6) / sqrt(esize + vsize + 1)

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
