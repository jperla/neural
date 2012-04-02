#!/usr/bin/env python
import numpy as np

from neurolib import T, sigmoid
import sample_images
import display_network
import softmax

def feedforward_autoencoder(theta, hidden_size, visible_size, data):
    """Accepts theta variable, hidden and visible size integers,
        and the data.
            Theta has shape (hidden_size*visible_size*2 
                                + hidden_size+visible_size,))
            data has shape (visible_size, num_examples).
        Returns activations on the hidden state.
            activations has shape (hidden_size, num_examples)
    """
    hv = hidden_size*visible_size
    assert theta.shape == (2*hv + hidden_size + visible_size,)
    W1 = theta[:hv].reshape(hidden_size, visible_size)
    b1 = theta[2*hv:2*hv+hidden_size]

    return sigmoid(np.dot(W1, data) + T(b1))

if __name__=='__main__':
    feature_model = np.load('mnist.model.npy')
    visible_size = 28*28
    hidden_size = 196
    num_classes = 5
    weight_decay = 1e-4
    max_iter = 400

    train_patches, train_labels = sample_images.get_mnist_data('../data/mnist.pkl.gz', lambda l: l <= 4, train=True, num_samples=100000)

    print 'will calculate training features...'
    train_activations = feedforward_autoencoder(feature_model, 
                                                hidden_size, 
                                                visible_size, 
                                                train_patches)
    assert train_activations.shape == (hidden_size, train_patches.shape[1])

    print 'will train classifier...'
    # train softmax classifier on autoencoded features
    trained = softmax.softmax_train(visible_size, 
                                    num_classes, 
                                    weight_decay, 
                                    train_activations,
                                    train_labels,
                                    max_iter)
    
    np.save('softmax.1to4.model', trained)
    display_network.display_network('softmax.1to4.png', trained.T)
 
    # use model to predict
    print 'will load test data...'
    test_patches, test_labels = sample_images.get_mnist_data('../data/mnist.pkl.gz', lambda l: l <= 4, train=False, num_samples=100000)

    print 'will compute test features...'
    test_activations = feedforward_autoencoder(feature_model, 
                                               hidden_size, 
                                               visible_size, 
                                               test_patches)
    assert test_activations.shape == (hidden_size, test_patches.shape[1])
    

    print 'will predict labels...'
    predicted_labels = softmax.softmax_predict(trained, test_patches)
    assert len(predicted_labels) == len(test_labels)
    print 'accuracy', 100 * np.mean(predicted_labels == test_labels)


