from functools import partial
from itertools import izip

import numpy as np

import scipy.optimize
import display_network
import sample_images
import sparse_autoencoder

if __name__ == '__main__':
    # Network Architecture 
    patch_size = (28,28)
    visible_size = patch_size[0] * patch_size[1]
    hidden_size = 196
    #hidden_size = 10

    # Training params
    weight_decay, sparsity_param, beta = 3e-3, 0.1, 3
    max_iter = 400	        # Maximum number of iterations of L-BFGS to run 

    # Get the data
    num_samples = 100
    num_samples = 100000

    # get digits from 5-9 from mnist data
    train, valid, test = sample_images.load_mnist_images('../data/mnist.pkl.gz')
    t = np.array([e for e,l in izip(train[0], train[1]) if l >= 5])
    v = np.array([e for e,l in izip(valid[0], valid[1]) if l >= 5])
    images = np.vstack([t, v]).T
    patches = images[:,:num_samples]
    assert patches.shape[0] == 784

    # set up L-BFGS args
    theta = sparse_autoencoder.initialize_params(hidden_size, visible_size)
    sae_cost = partial(sparse_autoencoder.cost,
                        visible_size=visible_size, 
                        hidden_size=hidden_size,
                        weight_decay=weight_decay,
                        beta=beta,
                        sparsity_param=sparsity_param,
                        data=patches)

    # Train!
    trained, cost, d = scipy.optimize.lbfgsb.fmin_l_bfgs_b(sae_cost, theta, 
                                                           maxfun=max_iter, 
                                                           m=100,
                                                           factr=10.0,
                                                           pgtol=1e-8,
                                                           iprint=1)
    # Save the trained weights
    W1, W2, b1, b2 = sparse_autoencoder.unflatten_params(trained,
                                                         hidden_size,
                                                         visible_size)
    display_network.display_network('mnist-features.png', W1.T)
    np.save('mnist.model', trained)


