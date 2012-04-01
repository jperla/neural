#!/usr/bin/env python
import functools

import numpy as np

import pca
import sample_images
import display_network

if __name__=='__main__':
    num_samples = 10000
    num_samples = 36

    m = sample_images.load_matlab_images('IMAGES_RAW.mat')
    patches = sample_images.sample(m, num_samples, size=(12,12), norm=None)

    display_network.display_network('raw-patches.png', patches)

    # ensure that patches have zero mean
    mean = np.mean(patches, axis=0)
    patches -= mean
    assert np.allclose(np.mean(patches, axis=0), np.zeros(patches.shape[1]))

    U, s, x_rot = pca.pca(patches)

    covar = pca.covariance(x_rot)
    display_network.array_to_file('covariance.png', covar)

    # percentage of variance
     # cumulative sum
    pov = np.array(functools.reduce(
                    lambda t,l: t+[t[-1]+l] if len(t) > 0 else [l],
                    s, 
                    [])
                  ) / np.sum(s)

    # first index greater than 99%
    k = np.min(np.where(pov >= 0.99))
    U, s, small_x_rot = pca.pca(patches, k=k)
    x_hat = np.dot(U[:,:k], small_x_rot)
    x_hat += mean
    display_network.display_network('99-reduced.png', x_hat)


    # first index greater than 90%
    k = np.min(np.where(pov >= 0.90))
    U, s, small_x_rot = pca.pca(patches, k=k)
    x_hat = np.dot(U[:,:k], small_x_rot)
    x_hat += mean
    display_network.display_network('90-reduced.png', x_hat)

    # first index greater than 50%
    k = np.min(np.where(pov >= 0.50))
    U, s, small_x_rot = pca.pca(patches, k=k)
    x_hat = np.dot(U[:,:k], small_x_rot)
    x_hat += mean
    display_network.display_network('50-reduced.png', x_hat)


    epsilon = 0.1
    _, _, x_pca_white = pca.pca_whiten(patches, epsilon=epsilon)
    c = pca.covariance(x_pca_white)
    display_network.array_to_file('pca_white_covariance.png', c)


    epsilon = 0.0
    _, _, x_pca_white = pca.pca_whiten(patches, epsilon=epsilon)
    c = pca.covariance(x_pca_white)
    display_network.array_to_file('pca_white_noreg_covariance.png', c)


    for epsilon in [1, 0.1, 0.01]:
        U, s, x_zca_white = pca.zca_whiten(patches, epsilon=epsilon)
        x_hat = x_zca_white
        display_network.display_network('zca_%s.png' % epsilon, x_hat)

