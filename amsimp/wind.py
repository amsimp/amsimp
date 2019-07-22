"""
AMSIMP Wind Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
import numpy as np
import pandas as pd
from amsimp.backend import Backend

# -----------------------------------------------------------------------------------------#


class Wind(Backend):
    """
	AMSIMP Wind Class - This class inherits the AMSIMP Backend Class.
	Calculates the Zonal and Meridional Wind. It will also create the vectors needed for projection
	onto the simulated globe.
	"""

    def geostrophic_wind(self):
        """
        Explain code here.
        """
        if self.detail_level < (5 ** (3 - 1)):
            raise Exception('detail_level must be greater than 3 in order to utilise this method.')
        
        # Distance of one degree of latitude (e.g. 0N - 1]\N/1S), measured in metres.
        lat_d = (2 * np.pi * self.a) / 360
        # Distance between latitude lines in the class method, Backend.latitude_lines().
        delta_y = (self.latitude_lines()[-1] - self.latitude_lines()[-2]) * lat_d

        gradient_geopotentialheight = []
        for Z in self.geopotential_height():
            southpole, northpole = np.split(Z, 2)

            southpole = np.flip(southpole)
            southpole = np.gradient(southpole)
            southpole = np.flip(southpole)

            northpole = np.gradient(northpole)

            delta_geoheight = np.concatenate((southpole, northpole))

            grad_geoheight = delta_geoheight / delta_y
            grad_geoheight = list(grad_geoheight)
            gradient_geopotentialheight.append(grad_geoheight)
        gradient_geopotentialheight = np.asarray(gradient_geopotentialheight)

        geostrophic_wind = (
            -(self.gravitational_acceleration() / self.coriolis_force())
            * gradient_geopotentialheight
        )

        return geostrophic_wind

    # -----------------------------------------------------------------------------------------#

    def contourf(self):
        """
		Explain code here.
		"""
        latitude, altitude = np.meshgrid(self.latitude_lines(), self.altitude_level())
        geostrophic_wind = self.geostrophic_wind()

        minimum = geostrophic_wind.min()
        maximum = geostrophic_wind.max()
        if minimum > -100 and maximum < 100:
            levels = np.linspace(minimum, maximum, 21)
        else:
            levels = np.linspace(-120, 70, 21)

        plt.contourf(latitude, altitude, geostrophic_wind, levels=levels)

        plt.set_cmap("jet")

        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Altitude (m)")
        plt.title("Geostrophic Wind in the Month of " + self.month.title())
        
        plt.figtext(0.99, 0.01, 'Note: Geostrophic balance does not hold near the equator.', horizontalalignment='right')
        plt.subplots_adjust(bottom = 0.135)

        colorbar = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=15)
        colorbar.locator = tick_locator
        colorbar.update_ticks()
        colorbar.set_label("Velocity ($\\frac{m}{s}$)")

        plt.show()
