#!/usr/bin/env python

import numpy as np
import display


def sigmoid(x): 
    """Accepts real.
        Returns real.
        Sigmoid function: range is [0,1].
        Similar to tanh function.
    """
    return 1.0 / (1.0 + np.exp(-x))


def sparse_autoencoder_cost(theta, visible_size, hidden_size, 
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
    hv = hidden_size * visible_size

    W1 = theta[1:hv].reshape(hidden_size, visible_size)
    W2 = theta[hv+1:2*hv].reshape(visible_size, hidden_size)
    b1 = theta[2*hv+1:2*hv+hidden_size]
    b2 = theta[2*hv+hidden_size+1:]

    # Cost and gradient variables (your code needs to compute these values). 

    # Here, we initialize them to zeros. 
    cost = 0
    W1grad = np.zeros(W1.shape)
    W2grad = np.zeros(W2.shape)
    b1grad = np.zeros(b1.shape)
    b2grad = np.zeros(b2.shape)

    num_data = data.shape[1]
    # do a feed forward pass
     # a2: (hidden_size, num_data)
    a2 = sigmoid(np.dot(W1, x) + b1.T)
     # a2: (visible_size, num_data)
    a3 = sigmoid(np.dot(W2, a2) + b2.T)
    assert a2.shape = (hidden_size, num_data)
    assert a3.shape = (visible_size, num_data)

    # compute the backprop
     # delta3: (visible_size, num_data)
    delta3 = -(data - a3) * (a3 * (1 - a3))
     # delta2: (hidden, num_data)
    delta2 = np.dot(W2.T, delta3) * (a2 * (1 - a2))

    W1grad = np.dot(delta2, data.T)
    W2grad = np.dot(delta3, a2.T)
    b1grad = delta2
    b2grad = delta3
    

