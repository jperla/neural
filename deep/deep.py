#!/usr/bin/env python
from functools import partial

import numpy as np
import scipy.optimize

import sample_images
import display_network
import softmax
import sparse_autoencoder

from selftaught import feedforward_autoencoder

if __name__=='__main__':
    #TODO: jperla: write this in a more general way! (N-layer network)
    #TODO: jperla: fine-tune!


    # load in unsupervised feature learning since it takes a while to compute
    l2_model = np.load('mnist.model.npy')

    visible_size = 28*28
    hidden_size = 196
    num_classes = 10
    softmax_weight_decay = 1e-4
    l3_weight_decay = 3e-3
    sparsity_param = 0.1
    beta = 3
    max_iter = 400
    num_samples = 1000000

    get_data = sample_images.get_mnist_data
    train_patches, train_labels = get_data('../data/mnist.pkl.gz', 
                                           train=True, 
                                           num_samples=num_samples)

    print 'will calculate l2 features...'
    l2_activations = feedforward_autoencoder(l2_model, 
                                                hidden_size, 
                                                visible_size, 
                                                train_patches)
    assert l2_activations.shape == (hidden_size, train_patches.shape[1])

    np.save('l2.0to9.model', l2_model)

    print 'will train layer 3 model'
    # set up L-BFGS args
    theta = sparse_autoencoder.initialize_params(hidden_size, hidden_size)
    sae_cost = partial(sparse_autoencoder.cost,
                        visible_size=hidden_size,
                        hidden_size=hidden_size,
                        weight_decay=l3_weight_decay,
                        beta=beta,
                        sparsity_param=sparsity_param,
                        data=l2_activations)

    # Train!
    l3_model, cost, d = scipy.optimize.lbfgsb.fmin_l_bfgs_b(sae_cost, theta, 
                                                            maxfun=max_iter, 
                                                            m=10,
                                                            factr=10.0,
                                                            pgtol=1e-8,
                                                            iprint=1)

    print 'will calculate l3 features...'
    l3_activations = feedforward_autoencoder(l3_model,
                                             hidden_size,
                                             hidden_size,
                                             l2_activations)
    assert l3_activations.shape == (hidden_size, l2_activations.shape[1])


    np.save('l3.0to9.model', l3_model)

    print 'will train classifier...'
    # train softmax classifier on autoencoded features
    classifier = softmax.softmax_train(hidden_size,
                                    num_classes,
                                    softmax_weight_decay,
                                    l3_activations,
                                    train_labels,
                                    max_iter)
    
    np.save('softmax.0to9.model', classifier)








    # use model to predict
    print 'will load test data...'
    test_patches, test_labels = get_data('../data/mnist.pkl.gz', 
                                         train=False, 
                                         num_samples=num_samples)

    print 'will compute test features...'
    test_l2_activations = feedforward_autoencoder(l2_model, 
                                                    hidden_size, 
                                                    visible_size, 
                                                    test_patches)
    assert test_l2_activations.shape == (hidden_size, test_patches.shape[1])

    print 'will compute test features...'
    test_l3_activations = feedforward_autoencoder(l3_model,
                                                    hidden_size,
                                                    hidden_size,
                                                    test_l2_activations)
    assert test_l3_activations.shape == (hidden_size, test_patches.shape[1])
    

    print 'will predict labels...'
    predicted_labels = softmax.softmax_predict(classifier, test_l3_activations)
    assert len(predicted_labels) == len(test_labels)
    print 'accuracy', 100 * np.mean(predicted_labels == test_labels)
    # 98.6 % accuracy!


