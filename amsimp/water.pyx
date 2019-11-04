#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
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
from cpython cimport bool

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

        Equation:
            e = 6.112 \* \exp((17.67 * T) / (T + 243.15))
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

        # Saturated water vapor pressure
        vapor_pressure = 6.112 * np.exp(
            (17.67 * temperature) / (temperature + 243.15)
        )

        # Add units of measurement.
        vapor_pressure *= self.units.hPa

        return vapor_pressure

    def mixing_ratio(self, pressure, vapor_pressure):
        """
        This method is solely utilised for integration in the
        amsimp.Water.precipitable_water() method. Please do not interact with
        the method directly.
        """
        y = (0.622 * vapor_pressure) / (pressure - vapor_pressure)
        return y

    cpdef np.ndarray precipitable_water(self, sum_altitude=True):
        """
        Generates a NumPy array of saturated precipitable water vapor.
        Precipitable water is the total atmospheric water vapor contained in a
        vertical column of unit cross-sectional area extending between any two
        specified levels. For a contour plot of this data, please use the
        amsimp.Water.contourf() method. If sum_altitude is equal to True, 
        it will sum the altitude component making the output 2-dimensional.

        Equation:
            PW = frac{1}{rho \* g} \* \int r dp
        """
        # Ensure sum_altitude is a boolean value.
        if not isinstance(sum_altitude, bool):
            raise Exception(
                "sum_altitude must be a boolean value. The value of which was: {}.".format(
                    sum_altitude
                )
            )

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

                    integration_term = quad(self.mixing_ratio, p1, p2, args=(e,))
                    Pwv_intergrationterm = integration_term[0]
                    pwv_alt.append(Pwv_intergrationterm)

                    k += 1

                if sum_altitude == False:
                    pwv_lat.append(pwv_alt)
                else:
                    P_wv = np.sum(pwv_alt)
                    pwv_lat.append(P_wv)                   

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
        cdef np.ndarray precipitable_water = self.precipitable_water(
            sum_altitude=True
        )

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

        # Add SALT.
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
        plt.close()
