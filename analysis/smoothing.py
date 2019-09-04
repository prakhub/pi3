#!/usr/bin/env python
# coding: utf-8

""" Provides smoothing functionality for the mean and covariance. """

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import numpy as np
# from scipy.signal import medfilt
from scipy.signal import convolve

def mean_moving_average(mean, n):
    """ Smooths the mean by using a moving average

    :param mean: Mean data to be smoothed
    :param n:    Size of the smoothing window

    """
    kernel = np.ones((n, 1)) / n
    return convolve(mean, kernel, mode="valid")

def cov_moving_average(cov, n):
    """ Smooths the covariance by using a moving average

    :param mean: Covarinace data to be smoothed
    :param n:    Size of the smoothing window

    """
    kernel = np.ones((n, 1, 1)) / n
    return convolve(cov, kernel, mode="valid")

