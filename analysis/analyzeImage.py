#!/usr/bin/env python
# coding: utf-8

""" Provides a function for calculating the mean and covariance
     for a list of or a single image"""

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import numpy as np


def analyze_image(data_list, scale=23.6 / 400):
    """ Analyzes image data and returns the mean and covariance

    Keyword arguments:
    data_list -- An arrray of images or a single image. If only a single image
                 is passed as an argument, it will be inserted into an array
                 of length 1.
    scale     -- Scaling factor between pixels and mm.

     """
    # if only a single image is passed, insert it into a array of length 1.
    if len(np.shape(data_list)) == 2:
        data_list = [np.asarray(data_list)]

    mean = []
    cov = []

    for data in data_list:

        # Calculate image dimensions
        sizex = np.shape(data)[1]
        sizey = np.shape(data)[0]

        # Generate coordinate matrix
        x = np.linspace(-scale * sizex, scale * sizex, sizex)
        y = np.linspace(-scale * sizey, scale * sizey, sizey)
        x, y = np.meshgrid(x, y)

        # Calculates the mean of a single image
        mean.append(np.asarray(
            (np.average(x, weights=data), np.average(y, weights=data))))
        
        # Calculates the covariance of a single image
        pos = np.stack((x.flatten(), y.flatten()))
        cov.append(np.cov(pos, aweights=data.flatten()))

    # Remove excess dimension if only a single image was passed as an argument
    return np.squeeze(np.asarray(mean)), np.squeeze(np.asarray(cov))
