import os, inspect
import random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import copy
import math

import numpy as np

from pysim.constants import RANDOM_POSITION

class CarPosition:

    def __init__(self, origin, calibration=False):
        self._origin = origin
        self._calibration = calibration

    def getNewPosition(self, trackpart):
        carsize = 0.205
        x = self._origin[0]
        y = self._origin[1]
        z = self._origin[2] + 0.03
        
        if RANDOM_POSITION:
            rnd_onemeter = random.random() * 1.0 - 0.5
            rnd_carwidth = random.random() * 0.5 - 0.25
            rnd_position = random.random() * 1.4 - 0.7  # ~ +/-40 grad
        else:
            rnd_onemeter = 0
            rnd_carwidth = 0
            rnd_position = 0

        trackdata = {
            1: [x + 2.9 - 0.7/2, y + 0.7, math.pi/2.0],                              # start position
            2: [x + 2.9 - 0.7 / 2 + rnd_carwidth,
                y + 6.5 / 2 + 1.0 + rnd_onemeter,
                math.pi/2.0 + rnd_position],
            3: [x + 2.9 / 2 + rnd_onemeter,
                y + 6.5 - 0.715 / 2 + rnd_carwidth,
                math.pi  + 0],
            4: [x + 0.7/2 + rnd_carwidth,
                y + 6.5 - carsize + rnd_onemeter,
                -math.pi/2.0 + rnd_position]
        }
        return trackdata[trackpart]

    def len(self):
        return 4
