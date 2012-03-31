import math

import Image
import numpy as np


def normalize_array(a, norm_max=255):
    """Accepts numpy array, and norm_max integer of max value.
       Returns an array normalized to go from 0 to norm_max.
    """
    c = a - np.min(a.flatten())
    c = c / np.max(c)
    centered = c * norm_max
    #min = np.min(a.flatten())
    #norm = np.max(a.flatten()) - min
    #normed = norm_max * a / norm
    #min = np.min(normed.flatten())
    #centered = normed - min
    return centered

def array_to_file(filename, a):
    """Accepts filename string,
        and numpy array.
        Saves png image representing array.
        Returns whether it was saved.
    """
    a = normalize_array(a)
    i = Image.fromarray(a.astype('uint8'))
    return i.save(filename)

def display_network(filename, images, padding=1):
    """Accepts filename string,
        2-d numpy array of images,
        and padding (default 1) number of black pixels between images.
        Each column of images is a filter. 

        This function visualizes filters in matrix images. 
        It will reshape each column into a square image and visualizes
            on each cell of the visualization panel. 

        Returns True on success.

        % TODO: jperla:
        % All other parameters are optional, usually you do not need to worry
        % about it.
        % opt_normalize: whether we need to normalize the filter so that all of
        % them can have similar contrast. Default value is true.
        % opt_graycolor: whether we use gray as the heat map. Default is true.
        % cols: how many columns are there in the display. Default value is the
        % squareroot of the number of columns in A.
        % opt_colmajor: you can switch convention to row major for A. In that
        % case, each row of A is a filter. Default value is false.
        warning off all
    """
    # first figure out the shape and size of everything
    s, n = images.shape
    d = int(math.sqrt(s))
    assert d * d == s, 'Images must be square'
    cols = int(math.sqrt(n))
    rows = int(n / cols) + (1 if n % cols > 0 else 0)

    # black background in output
    p = padding
    output = np.zeros((p + rows * (d + p), p + cols * (d + p)))
    output += np.min(images.flatten())
    # then fill in the output
    for i in xrange(n):
        r,c = int(i / cols), i % cols
        image = images[:,i]
        image.shape = (d,d)
        x,y = (r*(d+p))+p, (c*(d+p))+p
        output[x:x+d,y:y+d] = image

    # and save it 
    return array_to_file(filename, output)

