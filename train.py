from functools import partial

import scipy.optimize
import display_network
import sample_images
import sparse_autoencoder


if __name__ == '__main__':
    # Network Architecture 
    patch_size = (8,8)
    visible_size = patch_size[0] * patch_size[1]
    hidden_size = 25
    #hidden_size = 3

    # Training params
    weight_decay, sparsity_param, beta = 0.0001, 0.01, 3
    #weight_decay, sparsity_param, beta = 0, 0.01, 0
    max_iter = 400	        # Maximum number of iterations of L-BFGS to run 

    # Get the samples
    num_samples = 10000
    #num_samples = 10
    images = sample_images.load_matlab_images('IMAGES.mat')
    patches = sample_images.sample(images, num_samples, patch_size)

    # set up L-BFGS args
    theta = sparse_autoencoder.initialize_params(hidden_size, visible_size)
    sae_cost = partial(sparse_autoencoder.cost,
                        visible_size=visible_size, 
                        hidden_size=hidden_size,
                        weight_decay = weight_decay,
                        beta=beta,
                        sparsity_param=sparsity_param,
                        data=patches)

    # Train!
    trained, cost, d = scipy.optimize.lbfgsb.fmin_l_bfgs_b(sae_cost, theta, 
                                                           maxfun=max_iter, 
                                                           m=1000,
                                                           factr=1.0,
                                                           pgtol=1e-100,
                                                           iprint=1)
    '''
    # numerical approximation (way too slow!)
    trained, cost, d = scipy.optimize.lbfgsb.fmin_l_bfgs_b(
                                            lambda x: sae_cost(x)[0], theta, 
                                                           approx_grad=True,
                                                           maxfun=max_iter,
                                                           m=1000,
                                                           iprint=1)
    '''

    # Save the trained weights
    W1, W2, b1, b2 = sparse_autoencoder.unflatten_params(trained, 
                                                         hidden_size, 
                                                         visible_size)
    display_network.display_network('weights.png', W1.T)


