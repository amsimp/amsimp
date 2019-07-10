"""
AMSIMP Precipitable Water Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import math
import numpy as np
from amsimp.backend import Backend
from amsimp.wind import Wind

# -----------------------------------------------------------------------------------------#


class Water(Backend):
    """
    This class is concerned with calculating how water (specially precipitable water) moves
    from point to point within the troposphere and the stratosphere. 
    """

    def vapor_pressure(self):
        """
        Explain code here.
        """
        vapor_pressure = []

        temp = self.temperature() - 273.15

        for t in temp:
            if t >= 0:
                e = 0.61121 * np.exp((18.678 - (t / 234.5)) * (t / (257.14 + t)))
            elif t < 0:
                e = 0.61115 * np.exp((23.036 - (t / 333.7)) * (t / (279.82 + t)))
            vapor_pressure.append(e)
        vapor_pressure = np.asarray(vapor_pressure)

        vapor_pressure *= 1000
        return vapor_pressure

    def water_density(self):
        """
        Explain code here.
        """
        R_w = 461.52

        water_density = self.vapor_pressure() / (R_w * self.temperature())

        return water_density

    def precipitable_water(self):
        """
        This class calculates the amount of precipitable water in the troposphere
        and stratosphere. Precipitable water is the total atmospheric water vapor
        contained in a vertical column of unit cross-sectional area extending
        between any two specified levels. Mathematically, if the vapor pressure 
        is the mixing ratio, then the precipitable water vapor, W, contained in
        a layer bounded is by pressures p1 and p2.
        """
        precipitable_water = []

        pass

    def motion(self):
        """
        Explain code here.
        """
