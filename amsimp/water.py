"""
AMSIMP Precipitable Water Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import numpy as np
from amsimp.backend import Backend
from scipy.integrate import quad
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import ticker

# -----------------------------------------------------------------------------------------#


class Water(Backend):
    """
    AMSIMP Water Class - This class is concerned with calculating how much precipitable
    water vapor is in the air at a given latitude. Considering stratospheric air can be
    approximated as dry, to a reasonable degree of accuracy, this class will only
    consider tropospheric air.
    """

    def vapor_pressure(self):
        """
        Saturated vapor pressure.
        """
        vapor_pressure = []

        temperature = self.temperature() - 273.15
        delta_z = self.altitude_level()[1]
        index_of_maxz = int(10000 / delta_z)

        n = 0
        while n <= index_of_maxz:
            e = []

            temp = temperature[n]
            for t in temp:
                if t >= 0:
                    ans = 0.61121 * np.exp((18.678 - (t / 234.5)) * (t / (257.14 + t)))
                elif t < 0:
                    ans = 0.61115 * np.exp((23.036 - (t / 333.7)) * (t / (279.82 + t)))
                e.append(ans)
            vapor_pressure.append(e)

            n += 1
        vapor_pressure = np.asarray(vapor_pressure)

        vapor_pressure *= 1000
        return vapor_pressure

    def precipitable_water(self):
        """
        Precipitable water is the total atmospheric water vapor
        contained in a vertical column of unit cross-sectional area extending
        between any two specified levels. For a contourf plot of this data,
        please use the amsimp.Water.contourf() method.
        """
        precipitable_water = []

        pressure = np.transpose(self.pressure())
        vapor_pressure = np.transpose(self.vapor_pressure())
        g = -self.g
        rho_w = 0.997

        def integration_eq(pressure, vapor_pressure):
            y = (0.622 * vapor_pressure) / (pressure - vapor_pressure)
            return y

        n = 0
        while n < len(pressure):
            intergration = []

            delta_z = self.altitude_level()[1]
            index_of_maxz = int(10000 / delta_z)

            k = 0
            while k <= index_of_maxz:
                p1 = pressure[n][k]
                p2 = pressure[n][k + 1]

                if k < index_of_maxz:
                    e = (vapor_pressure[n][k] + vapor_pressure[n][k + 1]) / 2
                else:
                    e = vapor_pressure[n][k]

                integration_term = quad(integration_eq, p1, p2, args=(e,))
                intergration.append(integration_term)

                k += 1

            P_wv = np.sum(intergration)
            precipitable_water.append(P_wv)

            n += 1

        precipitable_water = np.asarray(np.transpose(precipitable_water))

        precipitable_water *= 1 / (rho_w * g)
        return precipitable_water

    def contourf(self):
        """
		Plots precipitable water on a contourf plot, with the axes being latitude and
        longitude. This plot is then layed on top of a EckertIII global projection.
        For the raw data, please use the amsimp.Water.precipitable_water() method.
		"""
        long = self.latitude_lines() * 2
        latitude, longitude = np.meshgrid(self.latitude_lines(), long)

        precipitable_water = []
        n = 0
        while n < len(longitude):
            precipitable_water.append(list(self.precipitable_water()))

            n += 1
        precipitable_water = np.asarray(precipitable_water)

        ax = plt.axes(projection=ccrs.EckertIII())

        ax.set_global()
        ax.coastlines()
        ax.gridlines()

        minimum = self.precipitable_water().min()
        maximum = self.precipitable_water().max()
        levels = np.linspace(minimum, maximum, 21)

        plt.contourf(
            longitude,
            latitude,
            precipitable_water,
            transform=ccrs.PlateCarree(),
            levels=levels,
        )

        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Longitude ($\lambda$)")
        plt.title("Precipitable Water in the Month of " + self.month.title())

        colorbar = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=15)
        colorbar.locator = tick_locator
        colorbar.update_ticks()
        colorbar.set_label("Precipitable Water (mm)")

        plt.show()
