#!/usr/bin/env python
# coding: utf-8

""" Provides an utility function to generate random data and random images."""
__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import numpy as np
from generateImage import generate_image


def generate_data(images, sizex=400, sizey=400):
    """ Generates a list of Gaussian images for testing the analysis software.

    :param images:  Amount of images to generate
    :param sizex :  Width of the image to be generated (default value = 400)
    :param sizey :  Height of the image to be generated (default value = 400)

    """
    im = []

    for i in range(images):

        # Linearly growing covariance
        sigmax = 0.2 * np.abs(np.sin(i / 3)) + 0.6
        sigmay = 0.2 * np.abs(np.cos(i / 3)) + 0.6

        # Mean on a circular trajectory with growing radius
        meanx = 0.15 * i * np.sin(i / 5) * 10
        meany = 0.02 * i * np.cos(i / 5)

        # Non zero diagonal entries rotate ellipse periodically
        diag = 0.01 * np.abs(np.sin(i / 3))

        m = [meanx, meany]
        cov = [[sigmax, diag], [diag, sigmay]]

        image = generate_image(mean=m, cov=cov, sizex=sizex, sizey=sizey)
        im.append(image)

    return im
