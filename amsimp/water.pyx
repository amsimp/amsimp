#cython: language_level=3
"""
AMSIMP Precipitable Water Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import numpy as np
from scipy.integrate import quad
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import ticker
from amsimp.wind cimport Wind
from amsimp.wind import Wind
cimport numpy as np

# -----------------------------------------------------------------------------------------#


cdef class Water(Wind):
    """
    AMSIMP Water Class - This class is concerned with calculating how much
    precipitable water vapor is in the air at a given latitude. Considering
    stratospheric air can be approximated as dry, to a reasonable degree of
    accuracy, this class will only consider tropospheric air.

    vapor_pressure ~ generates a NumPy array of saturated vapor pressure.
    precipitable_water ~ generates a NumPy array of saturated precipitable water
    vapor.
    water_contourf ~ generates a precipitable water vapor contour plot, and
    overlays that said plot onto a EckertIII projection of the Earth.
    """

    cpdef np.ndarray vapor_pressure(self):
        """
        Generates a NumPy array of saturated vapor pressure. Vapor pressure, in
        meteorology, is the partial pressure of water vapor. The partial
        pressure of water vapor is the pressure that water vapor contributes
        to the total atmospheric pressure.
        """
        # Ensures that the detail_level must be higher than 2 in order to utilise this method.
        if self.detail_level < (5 ** (3 - 1)):
            raise Exception(
                "detail_level must be greater than 2 in order to utilise this method."
            )

        cdef list list_vaporpressure = []

        # Convert temperature in Kelvin to degrees centigrade.
        cdef np.ndarray temperature = self.temperature() - 273.15

        # Troposphere boundary line.
        cdef float delta_z = self.altitude_level()[1]
        cdef float index_of_maxz = int(np.floor(self.troposphere_boundaryline()[0] / delta_z))

        # Calculations.
        cdef np.ndarray temp
        cdef list e
        cdef float t
        cdef int n
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
            list_vaporpressure.append(e)

            n += 1
        vapor_pressure = np.asarray(list_vaporpressure)

        # Convert from kPa to hPa.
        vapor_pressure *= 10
        
        # Convert from hPa to Pa.
        vapor_pressure *= 100

        return vapor_pressure

    def integration_eq(self, pressure, vapor_pressure):
        """
        This method is solely utilised for integration in the
        amsimp.Water.precipitable_water() method. Please do not interact with
        the method directly.
        """
        y = (0.622 * vapor_pressure) / (pressure - vapor_pressure)
        return y

    cpdef np.ndarray precipitable_water(self):
        """
        Generates a NumPy array of saturated precipitable water vapor.
        Precipitable water is the total atmospheric water vapor contained in a
        vertical column of unit cross-sectional area extending between any two
        specified levels. For a contour plot of this data, please use the
        amsimp.Water.contourf() method.
        """
        cdef list list_precipitablewater = []

        # Define variables.
        cdef float delta_z = self.altitude_level()[1]
        cdef float index_of_maxz = int(np.floor(self.troposphere_boundaryline()[0] / delta_z))

        cdef np.ndarray pressure = np.transpose(self.pressure()[0:index_of_maxz] / 100)
        cdef np.ndarray vapor_pressure = np.transpose(self.vapor_pressure() / 100)
        cdef float g = -self.g
        cdef float rho_w = 997

        # Integrate the mixing ratio with respect to pressure between the
        # pressure boundaries of p1, and p2.
        cdef list intergration
        cdef tuple integration_term
        cdef float p1, p2, e, P_wv, Pwv_intergrationterm
        cdef int n, k
        n = 0
        while n < len(pressure):
            intergration = []

            k = 0
            while k < (len(pressure[0]) - 1):
                p1 = pressure[n][k]
                p2 = pressure[n][k + 1]
                e = (vapor_pressure[n][k] + vapor_pressure[n][k + 1]) / 2

                integration_term = quad(self.integration_eq, p1, p2, args=(e,))
                Pwv_intergrationterm = integration_term[0]
                intergration.append(integration_term)

                k += 1

            P_wv = np.sum(intergration)
            list_precipitablewater.append(P_wv)

            n += 1

        precipitable_water = np.asarray(np.transpose(list_precipitablewater))

        # Multiplication term by integration.
        precipitable_water *= 1 / (rho_w * g)

        return precipitable_water

    def water_contourf(self):
        """
        Plots precipitable water on a contour plot, with the axes being
        latitude and longitude. This plot is then layed on top of a EckertIII
        global projection. For the raw data, please use the
        amsimp.Water.precipitable_water() method.
        """
        # Defines the axes.
        long = self.latitude_lines() * 2
        latitude, longitude = np.meshgrid(self.latitude_lines(), long)

        # Define the data.
        precipitable_water = []
        P_wv = self.precipitable_water()
        n = 0
        while n < len(longitude):
            precipitable_water.append(list(P_wv))

            n += 1
        precipitable_water = np.asarray(precipitable_water)

        # EckertIII projection details.
        ax = plt.axes(projection=ccrs.EckertIII())
        ax.set_global()
        ax.coastlines()
        ax.gridlines()

        # Contourf plotting.
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

        # Adds SALT to the graph.
        if self.future:
            month = self.next_month.title()
        else:
            month = self.month.title()

        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Longitude ($\lambda$)")
        plt.title("Precipitable Water in the Month of " + month)

        # Colorbar creation.
        colorbar = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=15)
        colorbar.locator = tick_locator
        colorbar.update_ticks()
        colorbar.set_label("Precipitable Water (mm)")

        plt.show()
