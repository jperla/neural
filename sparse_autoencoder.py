#!/usr/bin/env python
import math

import numpy as np

def cost(theta, visible_size, hidden_size,
         weight_decay, sparsity_param, beta, data):
    """
    % visible_size: the number of input units (probably 64) 
    % hidden_size: the number of hidden units (probably 25) 
    % lambda: weight decay parameter
    % sparsityParam: The desired average activation for the hidden units (denoted in the lecture
    %                           notes by the greek alphabet rho, which looks like a lower-case "p").
    % beta: weight of sparsity penalty term
    % data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
    
    % The input theta is a vector (because minFunc expects the parameters to be a vector). 
    % We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
    % follows the notation convention of the lecture notes. 
    """
    W1, W2, b1, b2 = unflatten_params(theta, hidden_size, visible_size)
    num_data = data.shape[1]

    # do a feed forward pass
    a2 = sigmoid(np.dot(W1, data) + T(b1))
    a3 = sigmoid(np.dot(W2, a2) + T(b2))
    assert a2.shape == (hidden_size, num_data)
    assert a3.shape == (visible_size, num_data)



    cost = 1.0 / num_data * ((0.5) * np.sum((a3 - data)**2))
    # add in weight decay
    cost += weight_decay / 2.0 * (np.sum(W1**2) + np.sum(W2**2))
    # add in sparsity parameter
    sparsity = np.sum(a2, axis=1) / float(num_data)
    assert sparsity.shape == (hidden_size,)
    s = sum(binary_KL_divergence(sparsity_param, p) for p in sparsity)
    cost += beta * s

    # delta3: Compute the backprop (product rule)
    delta3 = -(data - a3) * a3 * (1 - a3)
    assert delta3.shape == (visible_size, num_data)
    # delta2: Compute the backprop (product rule)
    # 1. calculate inner derivative
    delta2 = np.dot(W2.T, delta3) 
    # 2. add in sparsity parameter
    delta2 += T(beta * ((-sparsity_param / sparsity) +
                                ((1 - sparsity_param) / (1 - sparsity))))
    # 3. multiply by outer derivative
    delta2 *= a2 * (1 - a2)
    assert delta2.shape == (hidden_size, num_data)

    # compute final gradient
    W1grad = np.dot(delta2, data.T) / float(num_data)
    W2grad = np.dot(delta3, a2.T) / float(num_data)
    # add weight decay
    W1grad += weight_decay * W1
    W2grad += weight_decay * W2


    b1grad = np.sum(delta2, axis=1) / float(num_data)
    b2grad = np.sum(delta3, axis=1) / float(num_data)
    assert W1grad.shape == W1.shape
    assert W2grad.shape == W2.shape
    assert b1grad.shape == b1.shape
    assert b2grad.shape == b2.shape
    
    grad = flatten_params(W1grad, W2grad, b1grad, b2grad)
    return cost, grad

def initialize_params(hidden_size, visible_size):
    """Accepts number of hidde states in sparse encoder,
            and number of input states in sparse encoder..
       Initialize parameters randomly based on layer sizes.
       Returns a new flat array of size 2*visisble_size + hidden_size
    """
    assert hidden_size < visible_size

    #we'll choose weights uniformly from the interval [-r, r]
    r  = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.rand(hidden_size, visible_size) * 2 * r - r
    W2 = np.random.rand(visible_size, hidden_size) * 2 * r - r

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(visible_size)

    """
    % Convert weights and bias gradients to the vector form.
    % This step will "unroll" (flatten and concatenate together) all 
    % your parameters into a vector, which can then be used with minFunc. 
    """
    #TODO: jperla: make this a function
    return flatten_params(W1, W2, b1, b2)

def flatten_params(*args):
    """Accepts a list of matrices.
        Flattens and concatenates the matrices.
        Returns a 1-d array.
    """
    return np.hstack([a.flatten() for a in args])

def unflatten_params(theta, hidden_size, visible_size):
    """Accepts flat 1-D vector theta.
        Pulls out the weight vectors and returns them for 
            sparse autoencoding.
    """
    #TODO: jperla: generalize
    hv = hidden_size * visible_size
    W1 = theta[:hv].reshape(hidden_size, visible_size)
    W2 = theta[hv:2*hv].reshape(visible_size, hidden_size)
    b1 = theta[2*hv:2*hv+hidden_size]
    b2 = theta[2*hv+hidden_size:]
    return W1, W2, b1, b2

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
    p1, p2 = float(p1), float(p2)
    return (p1 * math.log(p1/p2)) + ((1 - p1) * math.log((1 - p1) / (1 - p2)))

