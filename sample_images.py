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

def load_matlab_images(matlab_filename):
    """Accepts a string.
        String points to file which is a matlab matrix .mat file.
        Loads the file and extracts the "IMAGES" key.
    """
    images = scipy.io.loadmat(matlab_filename)['IMAGES']
    return images

def sample(images, num_samples, size=(8,8)):
    """Accepts an array of images.
        images.ndim = (xdim, ydim, num_images)
        Also accepts the size of the sample, a 2-tuple, (xdim, ydim).
       Returns an array of flattened images.
            Will be a (size[0]*size[1]) x num_samples size array.
    """
    return np.array([random_sample(images, size)
                      for i in xrange(num_samples)]).T
    
    
if __name__=='__main__':
    num_samples = 10000
    images = load_matlab_images('IMAGES.mat')
    samples = sample(images, num_samples, (8,8))

    subset = samples[:, :200]
    try: 
        display.display_network('samples.png', subset)
    except Exception, e:
        print e
        import pdb; pdb.post_mortem()

