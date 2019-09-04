#!/usr/bin/env python
# coding: utf-8

""" Provides functions to export mean and covarinace data in CSV format. """

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'


import csv
import numpy as np
from scipy.stats import linregress


def export_mean(mean, filename, z_start=0, z_end=1):
    """ Exports the beam center position in x and y troughout
     the beamline into a csv.

    Keyword arguments:
    mean     -- Array containing the mean data
    filename -- Filename of the csv to be created
    z_start  -- Starting position of the screen in mm
    z_end    -- End position of the screen in mm
     """
    x = np.transpose(mean)[0]
    y = np.transpose(mean)[1]

    z = np.linspace(z_start, z_end, len(mean))

    with open(filename, mode='w') as file:
        wr = csv.writer(file, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
        wr.writerow(["z in mm", "x in mm", "y in mm"])
        for i in range(len(mean)):
            wr.writerow([z[i], x[i], y[i]])


def export_stddev(cov, filename, z_start=0, z_end=1):
    """ Exports the standard deviation in x and y troughout
     the beamline into a csv.

    Keyword arguments:
    cov      -- Array containing the covariance data used to calculate the std. dev.
    filename -- Filename of the csv to be created
    z_start  -- Starting position of the screen in mm
    z_end    -- End position of the screen in mm
     """
    # Calculate std. dev in x and y (std. dev. = sqrt(variance))
    sigmas = np.sqrt([cov.T[0][0], cov.T[1][1]]).T

    z = np.linspace(z_start, z_end, len(sigmas))

    with open(filename, mode='w') as file:
        wr = csv.writer(file, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)

        for i in range(len(sigmas)):
            wr.writerow([z[i], sigmas[i][0], sigmas[i][1]])


def export_cov(cov, filename, z_start=0, z_end=1):
    """ Exports the covariance matrix troughout the beamline into a csv.

    Keyword arguments:
    cov      -- Array containing the covariance data
    filename -- Filename of the csv to be created
    z_start  -- Starting position of the screen in mm
    z_end    -- End position of the screen in mm
     """
    z = np.linspace(z_start, z_end, len(cov))

    with open(filename, mode='w') as file:
        wr = csv.writer(file, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
        wr.writerow(["z in mm", "sigma xx in mm2", "sigma xy in mm2",
                     "ysigma yx in mm2", "sigma yy in mm2"])

        for i in range(len(cov)):
            wr.writerow([z[i], cov[i][0][0], cov[i][0][1], cov[i][1][0], cov[i][1][1]])


def export_mean_deviation(mean, filename, z_start=0, z_end=1):
    """ Exports the deviation of the mean from a linear fit at
     every measured position troughout the MBL

    Keyword arguments:
    mean     -- Array containing the mean data
    filename -- Filename of the csv to be created
    z_start  -- Starting position of the screen in mm
    z_end    -- End position of the screen in mm
     """
    z = np.linspace(z_start, z_end, len(mean))
    x = np.transpose(mean)[0]
    y = np.transpose(mean)[1]

    # create an linear fit of the mean
    slope, intercept, r_value, p_value, std_err = linregress(z, x)
    x_line = slope * z + intercept

    slope, intercept, r_value, p_value, std_err = linregress(z, y)
    y_line = slope * z + intercept

    with open(filename, mode='w') as file:
        wr = csv.writer(file, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)

        for i in range(len(mean)):
            wr.writerow([z[i], x[i] - x_line[i], y[i] - y_line[i]])


def export_surface(c, filename):
    """ Exports the surface points of a CubicFit to a csv

    Arguments:
    c        -- CubicFit to be exported
    filename -- Filename of the csv to be created

    """
    with open(filename, mode='w') as file:
        wr = csv.writer(file, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)

        X, Y, Z = c.generate_surface_data()

        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                wr.writerow([X[i][j], Y[i][j], Z[i][j]])


def CubicFit_parameters(c, filename):
    """ Exports the deviation of the mean from a linear fit at
     every measured position troughout the MBL

    Keyword arguments:
    c        -- CubicFit containing the parameters to be exported
    filename -- Filename of the csv to be created
     """

    with open(filename, mode='w') as file:
        wr = csv.writer(file, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)

        wr.writerow(["z in mm", "x in mm", "y in mm",
                     "a in mm", "b in mm", "angle in deg"])
        for i in range(len(c.s)):
            wr.writerow([c.s[i], c.xInterpolation(c.s[i]), c.yInterpolation(c.s[i]), c.aInterpolation(c.s[i]), c.bInterpolation(c.s[i]), c.angleInterpolation(c.s[i])])
