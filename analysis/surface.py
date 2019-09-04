#!/usr/bin/env python
# coding: utf-8

""" Provides a class to generate a 3D model from beam data """

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import numpy as np
from scipy.interpolate import CubicSpline


class CubicFit():
    """ Stores a 3D parameterization of the beam along the MBL.

        All parameters of an ellipse (center position and
        major and minor half-axis) are stored as a
        smooth fit and can be evaluated at any point inside the MBL
   """

    def fit(self, mean, cov, z_start=0, z_end=1):
        """ Fits an ellipse parameterization to beam data (mean,cov)

        :param mean:    Beam center positions to be fitted
        :param cov:     Covariance matrices to be fitted
        :param z_start: Starting position of the screen (default value = 0)
        :param z_end:   End position of the screen (default value = 1)
        """

        # convert to numpy arrays
        mean = np.asarray(mean)
        cov = np.asarray(cov)

        # Compute ellipse paramters from the eigenvalues of the covariance
        # matrix.
        # Take the sqrt of eigenvalues, because covariance is standard
        # deviation squared
        #eigenvals = np.sqrt(eigenvals)
        s = np.linspace(z_start, z_end, len(mean))  # z
        a = np.sqrt(cov.T[0][0])  # Major axis (a)
        b = np.sqrt(cov.T[1][1])  # Minor Axis (b)
        x = mean.T[0]    # X Offset of each ellipse
        y = mean.T[1]   # Y Offset of each ellipse
        self.s = s

        # Reverse order of data if monitor moved in negative z direction
        # This is necessary because CubicSpline() requires a monotonically
        # increasing x axis
        if z_start > z_end:
            s = np.flip(s)
            a = np.flip(a)
            b = np.flip(b)
            x = np.flip(x)
            y = np.flip(y)

        self.s = s
        # Fit data with cubic splines
        self.aInterpolation = CubicSpline(s, a)
        self.bInterpolation = CubicSpline(s, b)
        self.xInterpolation = CubicSpline(s, x)
        self.yInterpolation = CubicSpline(s, y)

    def generate_surface_data(self, N=100):
        """ Generates points on the surface of the fit to be used in a 3D plot.

        :param N: Amount of points used for creating data.

        """
        # TODO: Check if already fitted
        thetaGrid = np.linspace(0, 2 * np.pi, N)
        zGrid = np.linspace(np.min(self.s), np.max(self.s), N)

        X = np.array([self.xInterpolation(z) + self.aInterpolation(z)
                      * np.cos(thetaGrid) for z in zGrid])
        Y = np.array([self.yInterpolation(z) + self.bInterpolation(z)
                      * np.sin(thetaGrid) for z in zGrid])
        Z = np.array([z * np.ones(np.shape(thetaGrid)) for z in zGrid])

        return X, Y, Z

    def generate_surface_data_centered(self, N=100):
        """ Generates points on the surface of the fit centered
             around the beam path.

        :param N: Amount of points used for creating data.
        """

        thetaGrid = np.linspace(0, 2 * np.pi, N)
        zGrid = np.linspace(np.min(self.s), np.max(self.s), N)

        X = np.array([self.aInterpolation(z)
                      * np.cos(thetaGrid) for z in zGrid])
        Y = np.array([self.bInterpolation(z)
                      * np.sin(thetaGrid) for z in zGrid])
        Z = np.array([z * np.ones(np.shape(thetaGrid)) for z in zGrid])

        return X, Y, Z

    def generate_ellipse_data(self, N=100):
        """ Generates points on the ellipse crosssections

        :param N: Amount of points to generate
        """

        thetaGrid = np.linspace(0, 2 * np.pi, N)

        X = []
        Y = []
        Z = []

        for z in self.s:
            X.append(self.xInterpolation(z) +
                     self.aInterpolation(z) * np.cos(thetaGrid))
            Y.append(self.yInterpolation(z) +
                     self.bInterpolation(z) * np.sin(thetaGrid))
            Z.append(z * np.ones(np.shape(thetaGrid)))
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        return X, Y, Z


class CubicFitRotated():
    """ Stores a 3D parameterization of the beam along the MBL.

        All parameters of an ellipse (center position,
        major and minor half-axis and rotation angle) are stored as a
        smooth fit and can be evaluated at any point inside the MBL
    """

    def fit(self, mean, cov, z_start=0, z_end=1):
        """ Fits an ellipse parameterization to beam data (mean,cov).

        :param mean:    Beam center positions to be fitted
        :param cov:     Covariance matrices to be fitted
        :param z_start: Starting position of the screen (default value = 0)
        :param z_end:   End position of the screen (default value = 1)
        """

        # convert to numpy arrays
        mean = np.asarray(mean)
        cov = np.asarray(cov)

        # Compute ellipse paramters from the eigenvalues of the covariance
        # matrix.
        eigenvals, eigenvecs = np.linalg.eig(cov)
        # Take the sqrt of eigenvalues, because covariance is standard
        # deviation squared
        eigenvals = np.sqrt(eigenvals)
        s = np.linspace(z_start, z_end, len(mean))
        a = eigenvals.T[0]  # Major axis (a)
        b = eigenvals.T[1]  # Minor Axis (b)
        x = mean.T[0]    # X Offset of each ellipse
        y = mean.T[1]   # Y Offset of each ellipse

        angles = np.zeros(len(mean))
        
        # Calculation of this rotation angle does not yet work correctly
        for k in range(len(mean)):
            if np.linalg.norm(eigenvecs[k][1]) > np.linalg.norm(eigenvecs[k][0]):
                angles[k] = np.arccos(np.dot(eigenvecs[k][1], np.asarray(
                    [0, 1])) / (np.linalg.norm(eigenvecs[k][1])))
        
            else:
                angles[k] = np.arccos(np.dot(eigenvecs[k][0], np.asarray(
                     [0, 1])) / (np.linalg.norm(eigenvecs[k][0])))

        self.angles = angles

        # Reverse order of data if monitor moved in negative z direction
        # This is necessary because CubicSpline() requires a monotonically
        # increasing x axis
        self.s = s

        if z_start > z_end:
            s = np.flip(s)
            self.s = s
            a = np.flip(a)
            b = np.flip(b)
            x = np.flip(x)
            y = np.flip(y)

        # Fit data with cubic splines
        self.aInterpolation = CubicSpline(s, a)
        self.bInterpolation = CubicSpline(s, b)
        self.xInterpolation = CubicSpline(s, x)
        self.yInterpolation = CubicSpline(s, y)
        self.angleInterpolation = CubicSpline(s, angles)

    def generate_surface_data(self, N=200):
        """ Generates points on the surface of the fit.

        :param N: Amount of points used for creating data.

        """
        # TODO: Check if already fitted
        thetaGrid = np.linspace(0, 2 * np.pi, N)
        zGrid = np.linspace(np.min(self.s), np.max(self.s), N)

        X = np.array([self.xInterpolation(z) + self.aInterpolation(z) * np.cos(thetaGrid) * np.cos(self.angleInterpolation(
            z)) - self.bInterpolation(z) * np.sin(thetaGrid) * np.sin(self.angleInterpolation(z)) for z in zGrid])
        Y = np.array([self.yInterpolation(z) + self.aInterpolation(z) * np.cos(thetaGrid) * np.sin(self.angleInterpolation(
            z)) + self.bInterpolation(z) * np.sin(thetaGrid) * np.cos(self.angleInterpolation(z)) for z in zGrid])

        Z = np.array([z * np.ones(np.shape(thetaGrid)) for z in zGrid])

        return X, Y, Z

    def generate_surface_data_centered(self, N=100):
        """ Generates points on the surface of the fit centered around the beam path

        :param N: amount of points used for creating data.

        """
        thetaGrid = np.linspace(0, 2 * np.pi, N)
        zGrid = np.linspace(np.min(self.s), np.max(self.s), N)

        # print(self.angleInterpolation(zGrid))

        X = np.array([self.aInterpolation(z) * np.cos(thetaGrid) * np.cos(self.angleInterpolation(
            z)) - self.bInterpolation(z) * np.sin(thetaGrid) * np.sin(self.angleInterpolation(z)) for z in zGrid])
        Y = np.array([self.aInterpolation(z) * np.cos(thetaGrid) * np.sin(self.angleInterpolation(
            z)) + self.bInterpolation(z) * np.sin(thetaGrid) * np.cos(self.angleInterpolation(z)) for z in zGrid])

        Z = np.array([z * np.ones(np.shape(thetaGrid)) for z in zGrid])

        return X, Y, Z

    def generate_ellipse_data(self, N=100):
        """ Gnerates points on the ellipse crosssections

        :param N: Amount of points to generate
        """

        thetaGrid = np.linspace(0, 2 * np.pi, N)

        X = []
        Y = []
        Z = []

        for z in self.s:
            X.append(self.xInterpolation(z) + self.aInterpolation(z) * np.cos(thetaGrid) * np.cos(
                self.angleInterpolation(z)) - self.bInterpolation(z) * np.sin(thetaGrid) * np.sin(self.angleInterpolation(z)))
            Y.append(self.yInterpolation(z) + self.aInterpolation(z) * np.cos(thetaGrid) * np.sin(
                self.angleInterpolation(z)) + self.bInterpolation(z) * np.sin(thetaGrid) * np.cos(self.angleInterpolation(z)))
            Z.append(z * np.ones(np.shape(thetaGrid)))
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        return X, Y, Z
