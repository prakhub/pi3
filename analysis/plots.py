#!/usr/bin/env python
# coding: utf-8

""" Provides plots for analyzing and displaying measured data """

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import numpy as np
import matplotlib.pyplot as plt
from analyzeImage import analyze_image
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D

# change default font size for plots
import matplotlib
font = {'size': 15}
matplotlib.rc('font', **font)


def plot_cov(cov, z_start=0, z_end=1):
    """ Calculates and plots the standard deviation in x and y by distance along the MBL.

    :param cov:     Array containg covariance data, used to calculate
                     the standard deviation
    :param z_start: Starting position of the screen in mm (Default value = 0)
    :param z_end:   End position of the screen in mm (Default value = 1)

    """

    # Calculate std. dev in x and y (std. dev. = sqrt(variance))
    sigmas = np.sqrt([cov.T[0][0], cov.T[1][1]])

    z = np.linspace(z_start, z_end, len(cov))

    # Plot standard deviation in x and y by distance
    fig, ax = plt.subplots()
    ax.set_title("Standard  deviation in x and y by z")
    ax.scatter(z, sigmas[0], c='r', label='x', s=5)
    ax.scatter(z, sigmas[1], c='b', label='y', s=5)
    ax.legend(loc='upper center')
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Standard deviation [mm]")
    ax.set_ylim(0, 1.1 * np.max(sigmas))
    return fig


def plot_beta(cov, z_start=0, z_end=1):
    """ Calculates and plots the beta function in x and y by distance along the MBL.

    :param cov:     Array containg covariance data, used to calculate
                     the standard deviation
    :param z_start: Starting position of the screen in mm (Default value = 0)
    :param z_end:   End position of the screen in mm (Default value = 1)

    """

    betas = [cov.T[0][0], cov.T[1][1]]
    z = np.linspace(z_start, z_end, len(cov))

    # Plot standard deviation in x and y by distance
    fig, ax = plt.subplots()
    ax.set_title("σ² in x and y by the distance z along the MBL")
    ax.scatter(z, betas[0], c='r', label='σx²', s=5)
    ax.scatter(z, betas[1], c='b', label='σy²', s=5)
    ax.legend(loc='upper center')
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("σ² [mm²]")
    #ax.set_ylim(0, 1.1 * np.max(betas))
    return fig


def plot_cov_eigenvalues(cov, z_start=0, z_end=1):
    """ Calculates and plots the eigenvalues of the covariance by distance along the MBL.

    :param cov:     Array containg covariance data, used to calculate the eigenvalues
    :param z_start: Starting position of the screen in mm (Default value = 0)
    :param z_end:   End position of the screen in mm (Default value = 1)

    """

    # Calculate eigenvalues, discard eigenvectors
    eigenvals, retval = np.linalg.eig(cov)

    z = np.linspace(z_start, z_end, len(cov))

    fig, ax = plt.subplots()
    ax.set_title("Eigenvalues of the covariance matrix by distance")
    ax.scatter(z, eigenvals.T[0], c='r', label='x', s=5)
    ax.scatter(z, eigenvals.T[1], c='b', label='y', s=5)
    ax.legend(loc='upper center')
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Eigenvalue [mm]")
    return fig


def plot_mean(mean, z_start=0, z_end=1):
    """ Plots the mean (beam center position) in x and y by distance along the MBL.
        The uncertainty of the beam center position is 0.3 mm.
    :param mean:    Array containg the mean
    :param z_start: Starting position of the screen in mm (Default value = 0)
    :param z_end:   End position of the screen in mm (Default value = 1)

    """

    x = np.transpose(mean)[0]
    y = np.transpose(mean)[1]

    z = np.linspace(z_start, z_end, len(mean))

    fig, ax = plt.subplots()
    ax.set_title("Beam center in x and y by the distance z along the MBL")

    #ax.scatter(z, x, c='r', label='x', s=5)
    #ax.scatter(z, y, c='b', label='y', s=5)

    ax.errorbar(z, x, c='r', label='x', yerr=0.3, xerr=1,
                elinewidth=0.5, fmt='o', errorevery=1, ms=2)
    ax.errorbar(z, y, c='b', label='y', yerr=0.3, xerr=1,
                elinewidth=0.5, fmt='o', errorevery=1, ms=2)

    ax.legend(loc='lower center')
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Center position [mm]")

    return fig


def plot_mean_single(mean, z_start=0, z_end=1, axis=0):
    """ Plots the mean (beam center position) in x by distance.

    :param mean: the array containg the mean (beam center position)
    :param z_start: Starting position of the screen in mm (Default value = 0)
    :param z_end:   End position of the screen in mm (Default value = 1)

    """
    z = np.linspace(z_start, z_end, len(mean))

    fig, ax = plt.subplots()

    if not axis:
        ax.set_title("Beam center in x by z")
        ax.errorbar(z, x, c='r', label='x', yerr=0.3)

    if axis:
        ax.set_title("Beam center in y by z")
        ax.errorbar(z, y, c='b', label='y', yerr=0.3)

    ax.legend(loc='upper center')
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Center position [mm]")

    return fig


def plot_mean_distance(mean, z_start=0, z_end=1):
    """ Calculates and plots the distance of the beam center to the origin.

    :param mean:    Array containg the mean (beam center position)
    :param z_start: Starting position of the screen in mm (Default value = 0)
    :param z_end:   End position of the screen in mm (Default value = 1)

    """

    x = np.transpose(mean)[0]
    y = np.transpose(mean)[1]
    dist = np.sqrt(np.square(x) + np.square(y))
    z = np.linspace(z_start, z_end, len(mean))

    fig, ax = plt.subplots()
    ax.set_title("Euclidean distance of the beam center to the origin")
    ax.scatter(z, dist, c='b', label='y', s=5)
    ax.legend(loc='upper center')
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Distance of the beam center to the origin")
    return fig


def plot_mean_regression(mean, z_start=0, z_end=1, axis=0):
    """ Calculates a linear regression for the beam center position in x or y
        by the distance along the MBL and plots it with the fitted data.

    :param mean:    Array containg the mean (beam center position)
    :param z_start: Starting position of the screen in mm (Default value = 0)
    :param z_end:   End position of the screen in mm (Default value = 1)

    """

    z = np.linspace(z_start, z_end, len(mean))
    x = np.transpose(mean)[axis]
    slope, intercept, r_value, p_value, std_err = linregress(z, x)
    line = slope * z + intercept

    fig, ax = plt.subplots()
    ax.set_title("Linear fit of beam distance in x or y by z")
    ax.plot(z, line, color='r')
    ax.scatter(z, x, s=5)

    return fig


def plot_mean_deviation(mean, z_start=0, z_end=1, axis=0):
    """ Plots the deviation of the beam center movement from a linear fit 
         in either x or y by the distance along the MBL.

    :param mean:    Array containg the mean (beam center position)
    :param z_start: Starting position of the screen in mm (Default value = 0)
    :param z_end:   End position of the screen in mm (Default value = 1)

    """
    z = np.linspace(z_start, z_end, len(mean))
    x = np.transpose(mean)[axis]
    slope, intercept, r_value, p_value, std_err = linregress(z, x)
    line = slope * z + intercept

    fig, ax = plt.subplots()
    ax.set_title("Deviation from linear fit of beam distance in x or y by z")
    ax.scatter(z, x - line)
    # ax.plot(z,x-line)

    # ax.plot(line*0.01,z)

    return fig


def plot_mean_euclidean(mean, z_start=0, z_end=1):
    """ Plots the deviation of the distance between the beam center and origin
         from a linear regression

    :param mean:    Array containg the mean (beam center position)
    :param z_start: Starting position of the screen in mm (Default value = 0)
    :param z_end:   End position of the screen in mm (Default value = 1)

    """
    z = np.linspace(z_start, z_end, len(mean))
    x = mean.T[0]
    slope, intercept, r_value, p_value, std_err = linregress(z, x)
    line = slope * z + intercept
    x = x - line

    y = mean.T[1]
    slope, intercept, r_value, p_value, std_err = linregress(z, y)
    line = slope * z + intercept
    y = y - line

    dist = np.sqrt(np.square(x) + np.square(y))

    fig, ax = plt.subplots()
    ax.set_title(
        "Deviation from linear fit by the distance z along the MBL")
    ax.scatter(z, dist)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Deviation from linear fit [mm]")
    # ax.plot(z,x-line)

    # ax.plot(line*0.01,z)

    return fig


def plot_CubicFit(c, scaling_factor=1.5):
    """ Plots the surface of a CubicFit

    :param c:               A fitted CubicFit 
    :param scaling_factor:  Limits in plot = max(x,y) * scaling_factor
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("Beam envelope along the MBL")
    # Plot interpolation surface
    X, Y, Z = c.generate_surface_data(150)
    # ax.scatter(X, Y,Z)
    ax.plot_surface(X, Z, Y, rcount=10, ccount=100)
    #ax.scatter(Z, X, Y, s=0.5)

    max = np.max((np.max(np.abs(X)), np.max(np.abs(Y))))

    ax.set_xlim3d(-scaling_factor * max, scaling_factor * max)
    ax.set_zlim3d(-scaling_factor * max, scaling_factor * max)

    #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1]))
    ax.set_xlabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.view_init(29, -45)

    return fig


def plot_CubicFit_along_beam(c, N=50, scaling_factor=1):
    """ Plots the surface of a CubicFit along the beam
        This means that the beam center position is assumed to be at the 
         coordinate origin for all positions along the MBL.

    :param c: A fitted CubicFit 
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("Envelope along the beam axis")
    # Plot interpolation surface
    X, Y, Z = c.generate_surface_data_centered(N)
    # ax.scatter(X, Y,Z)
    ax.plot_surface(X, Z, Y, linewidth=0, rstride=1,
                    cstride=1, color=(0.352, 0.376, .99))

    #ax.scatter(Z, X, Y, s=0.5)

    max = np.max((np.max(np.abs(X)), np.max(np.abs(Y))))
    ax.set_xlim3d(-scaling_factor * max, scaling_factor * max)
    ax.set_zlim3d(-scaling_factor * max, scaling_factor * max)

    ax.set_xlabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.view_init(29, -45)

    return fig


def plot_CubicFit_ellipses(c):
    """ Plots the cross section ellipses of a CubicFit

    :param c: A fitted CubicFit

    """

    X, Y, Z = c.generate_ellipse_data(100)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("Beam envelope inside the beamline")

    # X,Y,Z = c.generate_ellipse_data(100)
    # print(np.shape(X),np.shape(Y),np.shape(Z))
    ax.scatter(Z, X, Y, lw=1, color='k', s=0.5)

    max = np.max(np.abs([X, Y]))
    # ax.set_ylim3d(-max,max)
    # ax.set_zlim3d(-max,max)
    ax.set_xlim3d(np.min(Z), np.max(Z))

    max = (np.max(Z) - np.min(Z)) / 2
    ax.set_ylim3d(-max, max)
    ax.set_zlim3d(-max, max)
    ax.set_ylabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_xlabel("z [mm]")
    ax.view_init(29, -45)
    return fig


def plot_CubicFit_angle(c):
    """ Plots the rotational angle of the ellipses of a CubicFit
         along the distance along the MBL

    :param c: A fitted CubicFit

    """
    fig, ax = plt.subplots()
    ax.set_title("Rotational angle of the beam ellipse by distance")

    ax.plot(c.s, c.angleInterpolation(c.s) / (2 * np.pi) * 360)

    ax.set_xlabel("distance [mm]")
    ax.set_ylabel("rotational angle [°]")
    return fig


def scroll_profile(x, images):
    """ Shows the x and y intensity profile of an individual image
         with fitted Gaussians and their parameters.

    :param x:      Index of the image to display
    :param images: List of images

    """

    im = images[x]
    data = np.asarray(im)

    # Normalize to maximum of x,y axis
    data = data / np.max(np.sum(data, axis=0))

    # Show image
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(im)

    # Calculate parameters for gaussian fit
    mean, cov = analyze_image(data)

    # Round mean, calculate std. dev. from variance and also round
    mean = np.around(mean, 2)
    # use np.abs, because we don't care about non diag. entries
    sigma = np.around(np.sqrt(np.abs(cov)), 2)

    # Plot x profile with fit parameters
    ax_x = fig.add_subplot(2, 2, 2)
    dat_x = np.sum(data, axis=0)

    ax_x.plot(dat_x)
    ax_x.set_xlabel("x in pixels")
    ax_x.set_ylabel("normalized luminosity")
    ax_x.set_title("x profile")

    plt.text(1, .1, r'$\mu=' + str(mean[0]) +
             ',\ \sigma $= ' + str(sigma[0, 0]))

    # Plot y profile with fit parameters
    ax_y = fig.add_subplot(2, 2, 3)
    dat_y = np.sum(data, axis=1)

    ax_y.plot(dat_y)
    ax_y.set_xlabel("y in pixels")
    ax_y.set_ylabel("normalized luminosity")
    ax_y.set_title("y profile")

    plt.text(1, .1, r'$\mu=' + str(mean[1]) +
             ',\ \sigma $= ' + str(sigma[1, 1]))

    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    return fig


# TODO: Find a method that works with preprocessing offset
def scroll_rotation(x, images, c, P):
    """ Shows an individual image with the fitted ellipse
         and eigenvectors of a CubicFit

    :param x:       Index of the image to be shown
    :param images:  List of images to be shown
    :param c:       A fitted CubicFit
    :parm P:        Preprocesser object used in preprocessing the images
                    Used to extract P.offset and P.image_size for scaling

    """
    fig, ax = plt.subplots()

    mean, cov = analyze_image(images[x])
    print("mean = " + str(mean))
    print("cov = " + str(cov))
    # convert to numpy arrays
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    # Compute ellipse paramters from the eigenvalues of the covariance matrix.
    eigenvals, eigenvecs = np.linalg.eig(cov)
    # Take the sqrt of eigenvalues, because covariance is standard deviation
    # squared
    eigenvals = np.sqrt(eigenvals)

    # Calculate rotation angle
    #angle = np.fmod(np.pi / 2 - np.arctan(eigenvecs[0][1] / eigenvecs[0][0]), np.pi)

    # Generate data of fitted ellipse
    X, Y, Z = c.generate_ellipse_data()
    X = X[x]
    Y = Y[x]

    # Size of the screen in mm to scale image
    screen_size_x = 23.6
    screen_size_y = 23.6

    # Correct image scaling if an offset was used in preprocessing
    if not P.offset == 0:
        screen_size_x = screen_size_x * (1 + P.offset / P.image_size[0])
        screen_size_y = screen_size_y * (1 + P.offset / P.image_size[1])

    image_to_show = np.flip(images[x], 0)
    plt.imshow(image_to_show, extent=[-screen_size_x, screen_size_x,
                                      -screen_size_y, screen_size_y], cmap='gray')

    plt.plot(X, Y, color='r')
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")

    # Plot eigenvectors originating from the middle of the ellipse (given by
    # the mean)
    plt.arrow(mean[0], mean[1], eigenvals[1] * eigenvecs[0][1], eigenvals[1] * eigenvecs[0][0],
              width=0.1, head_width=0.5, color='r')
    plt.arrow(mean[0], mean[1], eigenvals[0] * eigenvecs[1][1], eigenvals[0] * eigenvecs[1][0],
              width=0.1, head_width=0.5, color='b')

    return fig
