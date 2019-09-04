#!/usr/bin/env python
# coding: utf-8

""" Simple script to display a live view of the camera"""

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import cv2

cap = cv2.VideoCapture(0)

while True:

    retval, frame = cap.read()

    cv2.imshow('frame', frame)
    	
    # Wait 1 ms to check if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
