import copy
import math

import numpy as np

class CarPosition:

    def __init__(self, origin, calibration=False):
        self._origin = origin
        self._calibration = calibration

    def getNewPosition(self, trackpart):
        carsize = 0.205
        x = self._origin[0]
        y = self._origin[1]
        z = self._origin[2] + 0.03
        
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
                -math.pi/2.0 + rnd_position],
            5: [0.95, 4.75, math.pi/2],
            6: [1.5, 0.4, 2*math.pi],
            7: [0.4, 1.3, 1.5*math.pi],
            8: [1.8, 2.5, 1.5*math.pi],
            9: [1.8, 2, math.pi],
            10: [2, 4.75, 1.5*math.pi],
            11: [0.95, 4.75, math.pi/2]
            
        }
        return trackdata[trackpart]

    def len(self):
        return 11
