#!/usr/bin/env python
# coding: utf-8

""" Provides pre-processing functions which convert videos into usable data """

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from natsort import natsorted


class Preprocessor(object):
    """Provides an object which stores parameters and functions providing
         preprocessing and data import capabilities.

        These parameters are:
            - cornerPoints for perspective transform
            - threshold value used in thresholding
            - image_size which defines the size of the processed images
            - offset which defines the padding of the processed images
            - color to extract (0 Blue, 1 Green, 2 Red)
            - radius of the circle mask used
    """

    def __init__(self,
                 cornerPoints=np.float32(
                     [[194, 120], [380, 130], [180, 307], [375, 317]]),
                 min_threshold=0, max_threshold=255, image_size=[400, 400],
                 offset=50, color=0, radius=400):

        self.cornerPoints = cornerPoints
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.image_size = image_size
        self.offset = offset
        self.color = color
        self.radius = 400
        
    def import_images(self, path, extension):
        """Imports images from a directory with a specific extension.

        :param path:      Directory where the images are stored
        :param extension: Extension (format) of the images

        """
        self.frames = []

        files = glob.glob(path + '/*.' + extension)
        # sort numerically (i.e. 1,2,11 instead of  1,11,2)
        files = natsorted(files)

        for x in files:
            self.frames.append(cv2.imread(x))
        print("Imported", len(self.frames), "frames.")

    def import_video(self, filepath):
        """ Imports frames from a video and stores them for preprocessing.

        :param filepath: Path to the video file

        """
        self.frames = []
        vidcap = cv2.VideoCapture(filepath)

        success, im = vidcap.read()
        c = 0
        while success:
            self.frames.append(im)
            success, im = vidcap.read()
            c += 1

    def show_image(self, image):
        """ Displays an image.

        :param image: Image to be displayed

        """

        # Convert image from column-major to row-major order if neccessary
        if len(np.shape(image)) == 2:
            image = np.flip(image, 0)

        plt.imshow(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        plt.show()

    def preprocess(self):
        """Uses saved parameters and frames to preprocess and store the data ."""

        x = self.image_size[0]
        y = self.image_size[1]
        offset = self.offset

        # Corner points in processed image, where frame will be transformed to
        pts = np.float32([[offset, offset], [offset + x, offset],
                          [offset, offset + y], [offset + x, offset + y]])

        # creates an perspective transform and applies it to the images,
        # in order to obtain a "top-down" view of the images
        M = cv2.getPerspectiveTransform(self.cornerPoints, pts)

        self.frames_processed = []
        for img in self.frames:
            self.frames_processed.append(cv2.warpPerspective(
                img, M, (y + 2 * offset, x + 2 * offset)))

        # Extract specified color plane if image is not grayscale
        if len(np.shape(self.frames_processed)) >= 4:
            for i in range(len(self.frames_processed)):
                self.frames_processed[i] = cv2.split(
                    self.frames_processed[i])[self.color]

        # Apply a min. and max. threshold
        for i in range(len(self.frames_processed)):
            retval, self.frames_processed[i] = cv2.threshold(
                self.frames_processed[i], self.min_threshold, 255, cv2.THRESH_TOZERO)
            retval, self.frames_processed[i] = cv2.threshold(
                self.frames_processed[i], self.max_threshold, 255, cv2.THRESH_TOZERO_INV)

        # Apply a circular mask to cut off parts not on the foil
        mask = np.zeros(shape=self.frames_processed[0].shape, dtype="uint8")
        center = (np.asarray(
            np.shape(self.frames_processed[0])) / 2).astype(int)
        circ = cv2.circle(img=mask, center=(center[0], center[
                          1]), radius=self.radius, color=[255, 255, 255], thickness=-1)

        for i in range(len(self.frames_processed)):
            self.frames_processed[i] = cv2.bitwise_and(
                src1=self.frames_processed[i], src2=circ)

        # convert from  row-major to column-major ordering
        for i in range(len(self.frames_processed)):
            #   self.frames_processed[i] = self.frames_processed[i].T
            self.frames_processed[i] = np.flip(self.frames_processed[i], 0)

    def save_images(self, images, path):
        """ Utility function to save an array of images to a directory.

        :param images: Images to be saved
        :param path:   Directory where the images will be saved

        """

        for i in range(len(images)):

            filename = path + str(i) + ".png"
            cv2.imwrite(filename, images[i])

    def read_metadata(self, path):
        """Reads a metadata file and applies it to the stored preprocessing parameters.

        :param path: Path to the directory the metadata file is stored in.

        """

        exec(open(path + '/metadata.txt').read())
        self.cornerPoints = self.points
        self.z_start = self.distance[0]
        self.z_end = self.distance[1]
        self.frames = self.frames[self.cutFrames[0]:self.cutFrames[1]]
