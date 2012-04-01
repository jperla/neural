#!/usr/bin/env python
import re
import math
import random

import numpy as np

import pca

def load(filename):
    """Accepts filename.
        Reads in line by line.
        Each line is a dimension.
        Each line has N floating point numbers in ascii
        Returns an array of (N, #lines/dimensions) shape.
    """
    lines = [l.strip('\r\n ') for l in open(filename, 'r').readlines()]
    lines = [l for l in lines if l != '']
    dims = [re.split(r'\s+', l) for l in lines]
    f = np.array([[float(f) for f in d] for d in dims])
    return f

def scatter(filename, data, lines=[]):
    """Accepts an array of 2xN data.
        Also accepts an array of 4-tuples representing lines.
            A line is (x1, y1, x2, y2), which draws a line between
                (x1, y1) and (x2, y2).
        Saves the scatterplot of points in raw-scatterplot.png.
        Plot 2d data.
    """
    import matplotlib.pyplot as plot
    plot.figure(random.randint(0, 10000000))
    plot.scatter(data[0], data[1], 20, 'b', 'o')
    plot.title(filename.split('.')[0])
    for line in lines:
        plot.plot([line[0], line[2]], [line[1], line[3]], '-')
    plot.savefig(filename)

def T(a):
    return a.reshape(len(a), 1)

x = load('pcaData.txt')


U, s, x_rot = pca.pca(x)
scatter('raw-scatterplot.png', x, [((0,0)+tuple(r)) for r in U.T])

scatter('x-rot.png', x_rot)

k = 1
U, s, xHat = pca.pca(x, k)
scatter('xHat.png', np.vstack([xHat, xHat]))

xPCAwhite = pca.pca_whiten(x)
scatter('xPCAwhite.png', xPCAwhite)

xZCAwhite = pca.zca_whiten(x)
scatter('xZCAwhite.png', xZCAwhite)

