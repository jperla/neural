from functools import partial

import numpy as np

import sample_images
import numerical_gradient
import sparse_autoencoder

from test_numerical_gradient import diff_grad

def test_sae_cost():
    patch_size = (3,3)
    visible_size = patch_size[0] * patch_size[1]
    hidden_size = 3
    weight_decay, sparsity_param, beta = 0, 0, 0

    num_samples = 10
    images = sample_images.load_matlab_images('IMAGES.mat')
    patches = sample_images.sample(images, num_samples, patch_size)

    sae_cost = partial(sparse_autoencoder.cost,
                            visible_size=visible_size, 
                            hidden_size=hidden_size,
                            weight_decay=weight_decay, 
                            sparsity_param=sparsity_param, 
                            beta=beta,
                            data=patches)

    theta = sparse_autoencoder.initialize_params()
    cost, grad = sae_cost(theta)
    ncost, ngrad = numerical_gradient.compute(theta, 
                                              lambda x: sae_cost(x)[0],
                                              epsilon=0.0001)

    print ncost, cost
    print ngrad, grad

    diff = diff_grad(ngrad, grad)
    print diff

    less_than = 1e-9
    assert diff < less_than

