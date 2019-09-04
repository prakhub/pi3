#!/usr/bin/env python
# coding: utf-8

""" Generates beam-like images from a gaussian distribution """
__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import numpy as np
from scipy.stats import multivariate_normal
from PIL import Image


def generate_image(mean=[0, 0], cov=[[1, 0], [0, 1]],
                   sizex=400, sizey=400, scale=23.6 / 400):
    """ Generates a beam-like image defined by a gaussian distribution
    Keyword arguments:
    mean  -- Mean of the gaussian distribution
    cov   -- Covariance matrix of the gaussian distribution.
             Needs to be symmetric and positive-semidefinite
    sizex -- Width of the image to be generated
    sizey -- Height of the image to be generated
    scale -- Scaling factor between pixels and mm

    """
    # Generate coordinate matrix
    x = np.linspace(-scale * sizex, scale * sizex, sizex)
    y = np.linspace(-scale * sizey, scale * sizey, sizey)
    x, y = np.meshgrid(x, y)

    pos = np.array([x.flatten(), y.flatten()]).T

    # Compute samples of probabilty density function
    z = multivariate_normal.pdf(pos, mean, cov)

    # Normalize to (0,255) and scale to [0,255] interval (for conversion to unit8)
    z = 255 * z / np.max(z)

    # Round to integer
    z = np.round(z)

    # Create image from array
    z = z.reshape((sizey, sizex))
    im = Image.fromarray(z.astype(np.uint8))

    return im
