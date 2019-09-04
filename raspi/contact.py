#!/usr/bin/env python
# coding: utf-8

""" Checks if the motor is at the limit by reading switch high/low"""

__author__ = 'Andreas Gsponer'
__license__ = 'MIT'

import RPi.GPIO as GPIO



class contactSensor ():
    def __init__(self, pin):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.IN,pull_up_down=GPIO.PUD_UP)
    def touches(self):
        """ Monitors if the contact sensor is depressed """
        return GPIO.input(self.pin)

c = contactSensor(13)
print(c.touches())

GPIO.cleanup()
