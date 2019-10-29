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
        if self.detail_level < 3:
            raise Exception(
                "detail_level must be greater than 2 in order to utilise this method."
            )
        
        # Average boundary line between the troposphere and the stratosphere.
        troposphere_boundaryline = self.troposphere_boundaryline()
        avg_tropstratline = np.mean(troposphere_boundaryline)
        idx_troposphereboundaryline = (
            np.abs(self.altitude_level().value - avg_tropstratline.value)
        ).argmin()

        # Convert temperature in Kelvin to degrees centigrade.
        cdef np.ndarray temperature = self.temperature().to(
            self.units.deg_C, equivalencies=self.units.temperature()
        )
        temperature = temperature[:, :, 0:idx_troposphereboundaryline].value

        # Arden Buck equations.
        cdef np.ndarray temp, t
        cdef list list_vaporpressure = []
        cdef list e_lat, e_alt
        cdef float c, e
        for temp in temperature:
            e_lat = []
            for t in temp:
                e_alt = []
                for c in t:
                    if c >= 0:
                        e = 0.61121 * np.exp((18.678 - (c / 234.5)) * (c / (257.14 + c)))
                    elif c < 0:
                        e = 0.61115 * np.exp((23.036 - (c / 333.7)) * (c / (279.82 + c)))
                    e_alt.append(e)
                e_lat.append(e_alt)
            list_vaporpressure.append(e_lat)
        vapor_pressure = np.asarray(list_vaporpressure)

        # Convert from kPa to hPa.
        vapor_pressure *= 10
        vapor_pressure *= self.units.hPa

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
        # Average boundary line between the troposphere and the stratosphere.
        troposphere_boundaryline = self.troposphere_boundaryline()
        avg_tropstratline = np.mean(troposphere_boundaryline)
        idx_troposphereboundaryline = (
            np.abs(self.altitude_level().value - avg_tropstratline.value)
        ).argmin()

        # Defining some variables.
        cdef np.ndarray pressure = self.pressure().to(self.units.Pa).value
        pressure = pressure[:, :, 0:idx_troposphereboundaryline]
        cdef np.ndarray vapor_pressure = self.vapor_pressure().to(self.units.Pa)
        vapor_pressure = vapor_pressure.value
        cdef float g = -self.g.value
        cdef float rho_w = 997

        # Integrate the mixing ratio with respect to pressure between the
        # pressure boundaries of p1, and p2.
        cdef list list_precipitablewater = []
        cdef list pwv_lat, pwv_alt
        cdef tuple integration_term
        cdef float p1, p2, e, P_wv, Pwv_intergrationterm
        cdef int len_pressurelong = len(pressure)
        cdef int len_pressurelat = len(pressure[0])
        cdef int len_pressurealt = len(pressure[0][0]) - 1
        cdef int i, n, k
        i = 0
        while i < len_pressurelong:
            pwv_lat = []

            n = 0
            while n < len_pressurelat:
                pwv_alt = []

                k = 0
                while k < len_pressurealt:
                    p1 = pressure[i][n][k]
                    p2 = pressure[i][n][k + 1]
                    e = (vapor_pressure[i][n][k] + vapor_pressure[i][n][k + 1]) / 2

                    integration_term = quad(self.integration_eq, p1, p2, args=(e,))
                    Pwv_intergrationterm = integration_term[0]
                    pwv_alt.append(Pwv_intergrationterm)

                    k += 1

                pwv_lat.append(pwv_alt)

                n += 1
            
            list_precipitablewater.append(pwv_lat)
            i += 1

        precipitable_water = np.asarray(list_precipitablewater)

        # Multiplication term by integration.
        precipitable_water *= 1 / (rho_w * g)

        precipitable_water *= self.units.m
        precipitable_water = precipitable_water.to(self.units.mm)
        return precipitable_water

    def water_contourf(self):
        """
        Plots precipitable water on a contour plot, with the axes being
        latitude and longitude. This plot is then layed on top of a EckertIII
        global projection. For the raw data, please use the
        amsimp.Water.precipitable_water() method.
        """
        # Defines the axes, and the data.
        latitude, longitude = np.meshgrid(self.latitude_lines(),
         self.longitude_lines()
        )
        
        cdef np.ndarray precipitable_water = self.precipitable_water()
        precipitable_water = precipitable_water.value
        cdef np.ndarray p_water, pwv
        cdef list list_precipitablewater = []
        cdef list pwv_lat
        cdef float p_wv
        for p_water in precipitable_water:
            pwv_lat = []
            for pwv in p_water:
                p_wv = np.sum(pwv)
                pwv_lat.append(p_wv)
            list_precipitablewater.append(pwv_lat)
        precipitable_water = np.asarray(list_precipitablewater)
        precipitable_water *= self.units.mm

        # EckertIII projection details.
        ax = plt.axes(projection=ccrs.EckertIII())
        ax.set_global()
        ax.coastlines()
        ax.gridlines()

        # Contourf plotting.
        minimum = precipitable_water.min()
        maximum = precipitable_water.max()
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
        plt.title("Precipitable Water ("
         + self.date.strftime("%d-%m-%Y") + ")"
        )

        # Colorbar creation.
        colorbar = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=15)
        colorbar.locator = tick_locator
        colorbar.update_ticks()
        colorbar.set_label("Precipitable Water (mm)")

        plt.show()
