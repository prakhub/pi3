#!/usr/bin/env python
# coding: utf-8

""" Reads and analyzes all videos in subdirectories,
     then saves all plots possible"""

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import os
import multiprocessing
from glob import glob

import matplotlib.pyplot as plt

from surface import CubicFitRotated
from preprocess import Preprocessor
from analyzeImage import analyze_image
import plots
import smoothing
import export


def generate_plots(path):
    """ Generates plots for all videos in a directory

    :param path: the directory to search for videos

    """
    videos = glob(path + '/*.mkv')
    print(path, len(videos), videos)

    if len(videos) == 0:
        return
    else:
        videos = videos[0]

    metadata_list = glob(path + '/metadata.txt')
    #print(path, len(metadata_list), metadata_list)

    if len(metadata_list) == 0:
        return 

    P = Preprocessor()
    P.import_video(str(videos))
    P.read_metadata(path)
    P.preprocess()
    Im = P.frames_processed
    if len(Im) == 0:
        print(len(Im))
        return

    z_start = P.z_start
    z_end = P.z_end

    mean, cov = analyze_image(Im)

    window_size = 10
    mean_smoothed = smoothing.mean_moving_average(mean, window_size)
    cov_smoothed = smoothing.cov_moving_average(cov, window_size)

    c = CubicFitRotated()
    c.fit(mean=mean_smoothed, cov=cov_smoothed, z_start=z_start, z_end=z_end)

    try:
        os.mkdir(path + '/analysis')
        path += '/analysis'
    except OSError:
        pass


    plots.plot_mean(mean, z_start, z_end).savefig(path + '/beam_center.png')
    plots.plot_beta(cov, z_start, z_end).savefig(path + '/sigma_squared.png')

    export.export_mean(mean = mean, filename = path + '/center.csv', z_start = z_start, z_end = z_end)
    export.export_cov(cov = cov, filename = path + '/cov.csv', z_start = z_start, z_end = z_end)

    plt.close('all')


filepath = '../measurements/cyclotron/main_measurements/cyclotron/'

directories = [x[0] for x in os.walk(filepath)]

pool = multiprocessing.Pool()
pool.map(generate_plots, directories)
