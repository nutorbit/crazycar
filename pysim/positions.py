import math


class CarPosition:

    def __init__(self, origin, calibration=False):
        self._origin = origin
        self._calibration = calibration

    def getNewPosition(self, trackpart):
        x = self._origin[0]
        y = self._origin[1]
        z = self._origin[2] + 0.03

        trackdata = {
            1: [2.9 - 0.7/2, 1.1, math.pi/2],                              # start position
            2: [x + 2.9 - 0.7 / 2, y + 6.5 / 2 + 1.0, math.pi/2.0],
            3: [x + 2.9 / 2, y + 6.5 - 0.715 / 2, math.pi  + 0],
            # 4: [x + 0.7/2, y + 6.5 - carsize, -math.pi/2.0],
            4: [0.35, 3.25, 0],
            5: [0.95, 4.75, math.pi/2],
            6: [1.5, 0.4, 2*math.pi],
            7: [0.4, 1.3, 1.5*math.pi],
            8: [1.8, 4, 1.5*math.pi],
            9: [1.8, 2, math.pi],
            10: [2, 4.75, -math.pi/2],
            11: [0.95, 4.75, math.pi/2],
            13: [1.8, 2, math.pi/-2]
        }

        return trackdata[trackpart]

    def len(self):
        return 11
