# install 'sudo pip3 install adafruit-circuitpython-servokit'
# see hat reference

import board
import busio
import adafruit_pca9685
from adafruit_servokit import ServoKit

# settings
PWM_FREQ = 60
PWM_DUTYCYCLE_MAX = 0xffff

# modules
m1a_ch = 0
m1b_ch = 1
m1v_ch = 2

m2a_ch = 3
m2b_ch = 4
m2v_ch = 5

m3a_ch = 6
m3b_ch = 7
m3v_ch = 8

# vibration intensities
m1v_vibration = 0.2
m2v_vibration = 0.4
m3v_vibration = 0.6

# motor angle ranges
m1a_range = 180
m1b_range = 180
m2a_range = 180
m2b_range = 180
m3a_range = 180
m3b_range = 180

#### main ####
i2c = busio.I2C(board.SCL, board.SDA)
hat = adafruit_pca9685.PCA9685(i2c)
hat.frequency(PWM_FREQ)
kit = ServoKit(channels=16)


def vibration_init():
    """ Vibration Init

    Sets vibration motors to predescribed levels of intensity.
    Levels are not (yet) tailored to represent specific frequencies.
    """
    kit.continuous_servo[m1v_ch] = m1v_vibration
    kit.continuous_servo[m2v_ch] = m2v_vibration
    kit.continuous_servo[m3v_ch] = m3v_vibration


def motor_init():
    """ Motor Init

    Sets controls ranges for each module's pan/tilt stage.
    """
    kit.servo[m1a_ch].actuation_rangle = m1a_range
    kit.servo[m1b_ch].actuation_rangle = m1b_range
    kit.servo[m2a_ch].actuation_rangle = m2a_range
    kit.servo[m2b_ch].actuation_rangle = m2b_range
    kit.servo[m3a_ch].actuation_rangle = m3a_range
    kit.servo[m3b_ch].actuation_rangle = m3b_range


def motor_update(m1_dev, m2_dev, m3_dev):
    """ Motor Update

    Takes positional deviation and updates the angular rotation of each module.
    """

    # not sure if will work
    kit.servo[m1a_ch].angle++;
