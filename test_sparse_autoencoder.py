from functools import partial

import numpy as np

import sample_images
import numerical_gradient
import sparse_autoencoder

from test_numerical_gradient import diff_grad

def test_sae_cost():
    patch_size = (8,8)
    visible_size = patch_size[0] * patch_size[1]
    hidden_size = 25
    weight_decay, sparsity_param, beta = 0, 0, 0

    num_samples = 50
    images = sample_images.load_matlab_images('IMAGES.mat')
    patches = sample_images.sample(images, num_samples, patch_size)

    sae_cost = partial(sparse_autoencoder.cost,
                            visible_size=visible_size, 
                            hidden_size=hidden_size,
                            weight_decay=weight_decay, 
                            sparsity_param=sparsity_param, 
                            beta=beta,
                            data=patches)

    theta = sparse_autoencoder.initialize_params(hidden_size, visible_size)
    cost, grad = sae_cost(theta)
    ngrad = numerical_gradient.compute(theta,
                                       lambda x: sae_cost(x)[0],
                                       epsilon=0.0001)
    print cost
    print ngrad.shape, grad.shape
    print ngrad, grad

    diff = diff_grad(ngrad, grad)
    print diff

    threshold = 1e-9
    assert diff < threshold


    grad[8] = 1000
    assert diff_grad(ngrad, grad) > threshold


