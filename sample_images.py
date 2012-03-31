#!/usr/bin/env python
import random

import numpy as np
import scipy.io

import display

def randrange(r):
    """Accepts an integer
        Returns a random integer from 0..r-1
    """
    return random.randint(0, r-1)


def random_sample(images, size=(8,8)):
    """Accepts an array of images.
        images.ndim = (xdim, ydim, num_images)
        Also accepts the size of the sample, a 2-tuple.
       Returns a flattened array of a random patch of a random image.
            Will be a (size[0]*size[1]) x 1 size array.
    """
    num_images = images.shape[2]
    image = images[:,:,randrange(num_images)]
    
    x,y = size
    mx, my = tuple(np.array(image.shape) - np.array(size))
    rx, ry = randrange(mx), randrange(my)
    patch = image[rx:rx+x,ry:ry+y]
    return patch.flatten()
    

num_samples = 10000
images = scipy.io.loadmat('IMAGES.mat')['IMAGES']
samples = np.array([random_sample(images, (8,8))
                        for i in xrange(num_samples)]).T


subset = samples[:, :200]
try: 
    display.display_network('samples.png', subset)
except Exception, e:
    print e
    import pdb; pdb.post_mortem()

