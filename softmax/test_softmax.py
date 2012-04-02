from functools import partial

import numpy as np

import softmax
from test_numerical_gradient import check_grad

input_size = 28 * 28 # Size of input vector (MNIST images are 28x28)
num_classes = 10     # Number of classes (MNIST images fall into 10 classes)
weight_decay = 1e-4 # Weight decay parameter


def test_softmax_cost():
    num_examples = 20

    # for testing
    for i in xrange(5):
        input_size = 8
        data = np.random.randn(input_size, num_examples)
        labels = np.random.randint(0, num_classes, num_examples)
        theta = 0.005 * np.random.randn(num_classes * input_size)

        sc = partial(softmax.softmax_cost, num_classes=num_classes,
                                input_size=input_size,
                                weight_decay=weight_decay,
                                data=data,
                                labels=labels)

        yield check_grad, sc, theta, 5e-9

