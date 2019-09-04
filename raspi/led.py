#!/usr/bin/env python
# coding: utf-8

""" Controls the light of the camera 
    
    Use with "python -i led.py"
"""

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'


import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

PIN_LED = 12
GPIO.setup(PIN_LED, GPIO.OUT)

p = GPIO.PWM(PIN_LED, 500)
p.start(20)

# GPIO.cleanup()
