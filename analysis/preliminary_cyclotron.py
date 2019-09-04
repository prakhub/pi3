#!/usr/bin/env python
# coding: utf-8

""" "Main" script used to analyze invidual measurements"""
__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import numpy as np

from analyzeImage import analyze_image
from surface import CubicFitRotated
from preprocess import Preprocessor

import smoothing
import plots


P = Preprocessor(min_threshold=45, max_threshold=230, offset=0)

# Drift space focused by the BTL
#path = "../measurements/cyclotron/first_cyclotron_test/focused/3"
#P.import_video(path + "/focused_330_to_0.mkv")

# Changing steering current of the BTL
#path = "../measurements/cyclotron/first_cyclotron_test/changing_steering_magnet/2"
#P.import_video(path + "/changing_steering_magnet_330_to_0_10_steps.mkv")

# MBL magnets enabled
path = "../measurements/cyclotron/first_cyclotron_test/mbl_magnet/2"
P.import_video(path + "/mbl_magnet_330_to_0.mkv")

# m.set_size_inches(12,7)
# m.savefig("../report/images/measurements/cyclotron_pre/focused_center.png",dpi=300,optimize=True)

P.read_metadata(path)
P.preprocess()
Im = P.frames_processed

z_start = P.z_start
z_end = P.z_end

mean, cov = analyze_image(Im)

window_size = 10

mean_smoothed = smoothing.mean_moving_average(mean, window_size)
cov_smoothed = smoothing.cov_moving_average(cov, window_size)


plots.plot_mean(mean, z_start=z_start, z_end=z_end).show()
c = CubicFitRotated()
c.fit(mean=mean_smoothed, cov=cov_smoothed, z_start=z_start, z_end=z_end)


deviations = True

if deviations:

    x = mean.T[0]
    y = mean.T[1]

    print("mean x varied by:" + str(np.max(x) - np.min(x)))
    print("mean y varied by:" + str(np.max(y) - np.min(y)))

    xc = np.sqrt(cov_smoothed.T[0][0])
    yc = np.sqrt(cov_smoothed.T[1][1])
    print("cov x varied by:" + str(np.max(xc) - np.min(xc)))
    print("cov y varied by:" + str(np.max(yc) - np.min(yc)))

    xc = cov_smoothed.T[0][0]
    yc = cov_smoothed.T[1][1]
    print("beta x varied by:" + str(np.max(xc) - np.min(xc)))
    print("beta y varied by:" + str(np.max(yc) - np.min(yc)))
