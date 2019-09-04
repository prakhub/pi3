#!/usr/bin/env python3
# coding: utf-8

#
# copyright 2016 psionprime as registered user of pololu.com support forums
# Application of the AMIS-30543 Micro-Stepping Motor Drive circuit (onsemi.com)
# update 2019-03-29 UniBE/LHEP/PL
# 
# Take <qty> picture shoots during move on a certain distance <mm>:
# $ sudo ./old_motor.py --dirctrl=0 --pwmf=1 --pwmj=0 --sm=8 --dist=<mm> --step=<qty> 2>&1 >/dev/null

import sys
if sys.version_info < (3, 5):
    raise "must use python 3.5 or greater"
else:
    print("- python interpreter >= 3.5")

import gpiozero
from spidev import SpiDev
from RPIO import PWM
import time
import random
import getopt
from subprocess import call, DEVNULL

def CmdLine():

    dirctrl = 0x80              # setup default parameter values
    pwmf, pwmj  = 0x08, 0x04    # 46 kHz, jitter on
    sm, mult = 0x40, 32         # sm = 8, corresponding mutiplication factor 
    dist, step = 1, 1           # 1 mm, in 1 step
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"",["help","dirctrl=","pwmf=","pwmj=","sm=","dist=","step="])
    
    except getopt.GetoptError:
        print('$ 181010-pass-args.py --<args>, see --help for details\n')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '--help':
            print('  Possible options are (default values first):')
            print('--dirctrl =   1: forward,             0: backward')
            print('--pwmf    =   1: 46 kHz,              0: 23 kHz')
            print('--pwmj    =   1: jitter on,           0: jitter off')
            print('--sm      =  xx: 1/xx ustep, with xx = 32, 16, 8 or 4')
            print('            chs: comp. half step,   uhs: uncomp.')
            print('            cfs: comp. full step,   ufs: uncomp.')
            print('--dist    =  xx: distance [mm]')
            print('--step    =  xx: step number')            
            sys.exit()
        elif opt in ("--dirctrl"):
            dirctrl = {'1': 0x80, '0': 0x00}.get(arg,0x80)
        elif opt in ("--pwmf"):
            pwmf = {'1': 0x08, '0': 0x00}.get(arg,0x08)
        elif opt in ("--pwmj"):
            pwmj = {'1': 0x04, '0': 0x00}.get(arg,0x04)
        elif opt in ("--sm"):
            sm = {
                '32':  0x00,
                '16':  0x20,
                '8':   0x40,
                '4':   0x60,
                'chs': 0x80,
                'uhs': 0xA0,
                'cfs': 0xC0,
                'ufs': 0xE0
                }.get(arg,0x40)     # default value is 0x40 (sm = 8)
            mult = {
                '32': 32,
                '16': 16,
                '8': 8,
                '4': 4,
                'chs': 2,
                'uhs': 2,
                'cfs': 1,
                'ufs': 1
                }.get(arg,32)
        elif opt in ("--dist"):
            dist = float(arg)
        elif opt in ("--step"):
            step = int(arg)
                        
    return [dirctrl, pwmf, pwmj, sm, mult, dist, step]


class AMIS30543_Controller():

    def __init__(self, DO_DIRECTION_PIN, DO_RESET_PIN, DI_FAULT_PIN, ARGS):

        self.REG = {                # AMIS-30543 Registers
            'WR':  0x00,
            'CR0': 0x01,
            'CR1': 0x02,
            'CR2': 0x03,
            'CR3': 0x09,
            'SR0': 0x04,
            'SR1': 0x05,
            'SR2': 0x06,
            'SR3': 0x07,
            'SR4': 0x0A
        }

        self.CMD = {                # AMIS-30543 Command constants
        'READ': 0x00,
        'WRITE':0x80
        }

        self.dirctrl = ARGS[0]
        self.pwmf = ARGS[1]
        self.pwmj = ARGS[2]
        self.sm = ARGS[3]
        self.mult = ARGS[4]
        self.dist = ARGS[5]
        self.step = ARGS[6]
    
        self.VAL =  {
            'WR':  0b00000000,                                          # no watchdog
            'CR0': 0b00010111 | self.sm,                                # & 2.7 A current limit
            'CR1': 0b00000000 | self.dirctrl | self.pwmf | self.pwmj,   # & step on rising edge & fast slopes
            'CR2': 0b00000000,                                          # motor off & no sleep & SLA gain @ 0.5 & SLA no transparent
            'CR3': 0b00000000#,                                         # no extended step mode
            #'dist': self.dist,
            #'step': self.step
        }

        # InitGPIO

        PWM.setup(5,0)              # 5 us pulse_incr, 0 delay_hw
        PWM.init_channel(0,3000)    # DMA channel 0, 3000 us subcycle time

        self.DO_RESET = gpiozero.DigitalOutputDevice(DO_RESET_PIN)
        self.DO_DIRECTION = gpiozero.DigitalOutputDevice(DO_DIRECTION_PIN)
        self.DI_NO_FAULT = gpiozero.DigitalInputDevice(DI_FAULT_PIN)

        self.spi = SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 1000000

        self.RegisterSet()


    def __del__(self):
        self.spi.close()


    def ResetStepper(self):
        self.DO_RESET.off()     # must be off for AMIS to see reset
        time.sleep(0.11)
        self.DO_RESET.on()
        time.sleep(0.11)
        self.DO_RESET.off()


    def RegisterDump(self):                 # to check stepper status
        print("\nAMIS-30543 Registers:")
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['WR'], 0])
        print(" WR = ", bin(resp[1]), " ", str(resp[1]))
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['CR0'], 0])
        print("CR0 = ", bin(resp[1]), " ", str(resp[1]))
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['CR1'], 0])
        print("CR1 = ", bin(resp[1]), " ", str(resp[1]))
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['CR2'], 0])
        print("CR2 = ", bin(resp[1]), " ", str(resp[1]))
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['CR3'], 0])
        print("CR3 = ", bin(resp[1]), " ", str(resp[1]))
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['SR0'], 0])
        print("SR0 = ", bin(resp[1]), " ", str(resp[1]))
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['SR1'], 0])
        print("SR1 = ", bin(resp[1]), " ", str(resp[1]))
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['SR2'], 0])
        print("SR2 = ", bin(resp[1]), " ", str(resp[1]))
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['SR3'], 0])
        print("SR3 = ", bin(resp[1]), " ", str(resp[1]))
        resp = self.spi.xfer2([self.CMD['READ'] | self.REG['SR4'], 0])
        print("SR4 = ", bin(resp[1]), " ", str(resp[1]))
        print("")


    def RegisterSet(self):
        self.ResetStepper()
 
        self.spi.writebytes([self.CMD['WRITE'] | self.REG['WR'], self.VAL['WR']])
        self.spi.writebytes([self.CMD['WRITE'] | self.REG['CR0'], self.VAL['CR0']])
        self.spi.writebytes([self.CMD['WRITE'] | self.REG['CR1'], self.VAL['CR1']])
        self.spi.writebytes([self.CMD['WRITE'] | self.REG['CR2'], self.VAL['CR2']])
        self.spi.writebytes([self.CMD['WRITE'] | self.REG['CR3'], self.VAL['CR3']])

        if self.spi.xfer2([self.CMD['READ'] | self.REG['WR'], 0])[1] != self.VAL['WR']:
            print("Writing or reading self.REG['WR'] failed; driver power might be off.")
            return False
        if self.spi.xfer2([self.CMD['READ'] | self.REG['CR0'], 0])[1] != self.VAL['CR0']:
            print("Writing or reading self.REG['CR0'] failed; driver power might be off.")
            return False
        if self.spi.xfer2([self.CMD['READ'] | self.REG['CR1'], 0])[1] != self.VAL['CR1']:
            print("Writing or reading self.REG['CR1'] failed; driver power might be off.")
            return False
        if self.spi.xfer2([self.CMD['READ'] | self.REG['CR2'], 0])[1] != self.VAL['CR2']:
            print("Writing or reading self.REG['CR2'] failed; driver power might be off.")
            return False
        if self.spi.xfer2([self.CMD['READ'] | self.REG['CR3'], 0])[1] != self.VAL['CR3']:
            print("Writing or reading self.REG['CR3'] failed; driver power might be off.")
            return False

        #self.RegisterDump()
        #print("RegisterSet Ok\n")
        return True


    def SetMotorEnable(self):

        self.spi.writebytes([self.CMD['WRITE'] | self.REG['CR2'], self.VAL['CR2'] | 0b10000000])

        if self.spi.xfer2([self.CMD['READ'] | self.REG['CR2'], 0])[1] != self.VAL['CR2'] | 0b10000000:
            print("Writing or reading self.REG['CR2'] failed; driver power might be off.")
            return False


    def SetMotorDisable(self):

        self.spi.writebytes([self.CMD['WRITE'] | self.REG['CR2'], self.VAL['CR2'] & 0b01111111])

        if self.spi.xfer2([self.CMD['READ'] | self.REG['CR2'], 0])[1] != self.VAL['CR2'] & 0b01111111:
            print("Writing or reading self.REG['CR2'] failed; driver power might be off.")
            return False

def clean():
    command = 'rm /var/picture-cache/*.jpg'
    call(command, shell=True)

def take_startup_picture():
    command = 'sudo -u pi3-mfs avconv -f video4linux2 -s 640x480 -i /dev/video0 -ss 0:0:2 -frames 1 /var/picture-cache/null.jpg'
    call(command, shell=True, stderr=DEVNULL, stdout=DEVNULL)

def take_picture(indice):
    command = 'sudo -u pi3-mfs avconv -f video4linux2 -s 640x480 -i /dev/video0 -frames 1 /var/picture-cache/' + str(indice) + '.jpg'
    call(command, shell=True, stderr=DEVNULL, stdout=DEVNULL)

print(CmdLine())

axis_X1 = AMIS30543_Controller(24, 27, 18, CmdLine())

#clean()
#take_startup_picture()
#clean()

for i in range(24):                     # DMA channel = 0 & GPIO25 & start at 25*5us*i & 10*5us width
    PWM.add_channel_pulse(0,25,25*i,10) # -> 50/125 us width pulse at 8kHz

# M6: 1.   mm thread - 6.25 um
# steps for dist : dist/thread * (200 * axis_X1.mult)
timeForDistance = axis_X1.dist * (200 * axis_X1.mult / 8000) # steps for dist / frequency
timeForStep = timeForDistance / axis_X1.step

#take_picture(0)
startTime = time.monotonic()

for i in range(axis_X1.step):
    axis_X1.SetMotorEnable()
    time.sleep(timeForStep)
    axis_X1.SetMotorDisable()
    #time.sleep(0.1)
    #take_picture(i+1)

deltaTime = time.monotonic() - startTime    
PWM.clear_channel_gpio(0, 25)
PWM.clear_channel(0)
PWM.cleanup()

print("\nStep Test took ", deltaTime, " secs")
print("\nSpeed: ", axis_X1.dist / deltaTime, " mm/sec\n")


