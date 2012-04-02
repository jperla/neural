from functools import partial

import numpy as np

import sample_images
import numerical_gradient
import sparse_autoencoder

from test_numerical_gradient import diff_grad, check_grad

patch_size = (8,8)
visible_size = patch_size[0] * patch_size[1]
#hidden_size = 25
hidden_size = 3
weight_decay, sparsity_param, beta = 0.0001, 0.01, 3
#weight_decay, sparsity_param, beta = 0, 0.01, 0

#num_samples = 10
num_samples = 100
images = sample_images.load_matlab_images('../data/IMAGES.mat')
patches = sample_images.sample(images, num_samples, patch_size)

base_sae_cost = partial(sparse_autoencoder.cost,
                        visible_size=visible_size, 
                        hidden_size=hidden_size,
                        sparsity_param=sparsity_param,
                        data=patches)

def test_sae_cost():
    threshold = 1e-9 * (num_samples / 50.0)
    theta = sparse_autoencoder.initialize_params(hidden_size, visible_size)

    sae_cost = partial(base_sae_cost, weight_decay=weight_decay, beta=beta)
    cost, grad, ngrad = check_grad(sae_cost, theta, threshold)


    # test that if gradient is wrong, we fail
    bad_grad = np.array(grad)
    bad_grad[2] = 1000
    assert diff_grad(ngrad, bad_grad) > threshold
    bad_grad2 = 2 * np.array(grad)
    assert diff_grad(ngrad, bad_grad2) > threshold

    # test that weight params actually do something
    if weight_decay > 0:
        noweight_sae_cost = partial(base_sae_cost, weight_decay=0, beta=beta)
        noweight_cost, noweight_grad, _ = check_grad(noweight_sae_cost, 
                                                    theta, threshold)
        print 'noweight cost:', noweight_cost
        diff = diff_grad(grad, noweight_grad)
        print 'noweight diff:', diff
        assert diff > threshold


    # test that sparsity works
    if beta > 0:
        nosparsity_sae_cost = partial(base_sae_cost,
                                      weight_decay=weight_decay,
                                      beta=0)
        nosparsity_cost, nosparsity_grad, _ = check_grad(nosparsity_sae_cost,
                                                         theta, threshold)
        print 'nosparsity cost:', nosparsity_cost
        diff = diff_grad(grad, nosparsity_grad)
        print 'nosparsity diff:', diff
        assert diff > threshold

