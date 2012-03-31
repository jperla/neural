from functools import partial

import numpy as np

import sample_images
import numerical_gradient
import sparse_autoencoder

from test_numerical_gradient import diff_grad

patch_size = (8,8)
visible_size = patch_size[0] * patch_size[1]
#hidden_size = 25
hidden_size = 3
weight_decay, sparsity_param, beta = 0.0001, 0.01, 3

#num_samples = 10
num_samples = 100
images = sample_images.load_matlab_images('IMAGES.mat')
patches = sample_images.sample(images, num_samples, patch_size)

def test_sae_cost():
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

    threshold = 1e-9 * (num_samples / 50.0)
    assert diff < threshold


    # test that if gradient is wrong, we fail
    grad[8] = 1000
    assert diff_grad(ngrad, grad) > threshold


