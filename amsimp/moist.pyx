#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
"""
AMSIMP Moist Thermodynamics Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import numpy as np
from scipy.integrate import quad
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
from matplotlib import ticker
from amsimp.backend cimport Backend
from amsimp.backend import Backend
cimport numpy as np
from cpython cimport bool

# -----------------------------------------------------------------------------------------#


cdef class Moist(Backend):
    """
    AMSIMP Moist Thermodynamics Class - This class is concerned with calculating how much
    precipitable water vapor is in the air at a given latitude. Considering
    stratospheric air can be approximated as dry, to a reasonable degree of
    accuracy, this class will only consider tropospheric air. (Change description).

    vapor_pressure ~ generates a NumPy array of saturated vapor pressure.
    precipitable_water ~ generates a NumPy array of saturated precipitable water
    vapor.
    water_contourf ~ generates a precipitable water vapor contour plot, and
    overlays that said plot onto a EckertIII projection of the Earth.
    """

    cpdef np.ndarray vapor_pressure(self):
        """
        Generates a NumPy array of vapor pressure. Vapor pressure, in
        meteorology, is the partial pressure of water vapor. The partial
        pressure of water vapor is the pressure that water vapor contributes
        to the total atmospheric pressure.

        Equation (Saturated Vapor Pressure):
            e_s = 6.112 \* \exp((17.67 * T) / (T + 243.15))

        Equation (Vapor Pressure):
            e = frac{r_h \* e_s}{100}
        """
        # Convert temperature in Kelvin to degrees centigrade.
        cdef np.ndarray temperature = self.temperature().value - 273.15

        # Saturated water vapor pressure
        sat_vapor_pressure = 0.61121 * np.exp(
            (
                18.678 - (temperature / 234.5)
            ) * (temperature / (257.14 + temperature)
            )
        )

        # Add units of measurement.
        sat_vapor_pressure *= self.units.kPa
        sat_vapor_pressure = sat_vapor_pressure.to(self.units.hPa)

        # Vapor pressure, accounting for relative humidity.
        vapor_pressure = self.relative_humidity().value * sat_vapor_pressure
        vapor_pressure /= 100

        return vapor_pressure

    cpdef np.ndarray virtual_temperature(self):
        """
        Explain here.
        """
        virtual_temperature = self.temperature() / (
            1 - (
            self.vapor_pressure() / self.pressure_surfaces(dim_3d=True)
            ) * (1 - 0.622)
        )

        return virtual_temperature

# -----------------------------------------------------------------------------------------#

    cpdef np.ndarray density(self):
        """
        Generates a NumPy array of atmospheric pressure utilising the Ideal
        Gas Law. The ideal gas equation is the equation of state for the
        atmosphere, and is defined as an equation relating temperature,
        pressure, and specific volume of a system in theromodynamic
        equilibrium.

        Equation:
            rho = frac{p}{R * T_v}

        The atmospheric density is the mass of the atmosphere per unit
        volume.
        """
        cdef np.ndarray pressure_surfaces = self.pressure_surfaces(dim_3d=True)

        density = pressure_surfaces / (self.virtual_temperature() * self.R)

        # Switch to the appropriate SI units.
        density = density.si
        return density

    cpdef np.ndarray exner_function(self):
        """
        Generates a NumPy array of the exner function. The Exner function can be
        viewed as non-dimensionalized pressure.

        Equation:
            \Pi = frac{T}{theta}
        """
        exner_function = self.virtual_temperature() / self.potential_temperature()

        return exner_function

    cpdef np.ndarray potential_temperature(self, moist=False):
        """
        Generates a NumPy array of potential temperature. The potential
        temperature of a parcel of fluid at pressure P is the temperature that
        the parcel would attain if adiabatically brought to a standard reference
        pressure

        Equation:
            theta = T \* (frac{P}{P_0}) ** (-R / c_p)
        """
        # Ensure moist is a boolean value.
        if not isinstance(moist, bool):
            raise Exception(
                "moist must be a boolean value. The value of moist was: {}".format(
                    moist
                )
            )

        # Determine whether to calculate wet-bulb potential temperature, or
        # potential temperature.
        cdef np.ndarray temperature
        if moist:
            temperature = self.virtual_temperature().value
        else:
            temperature = self.temperature().value

        cdef np.ndarray pressure_surfaces = self.pressure_surfaces(dim_3d=True).value
        cdef float R = self.R.value
        cdef float c_p = self.c_p.value

        cdef list list_potentialtemperature = []
        cdef int n = 0
        cdef int len_psurfaces = len(pressure_surfaces)
        while n < len_psurfaces:
            theta = temperature[n] * (
                (pressure_surfaces[n] / pressure_surfaces[0]) ** (-R / c_p)
            )
            
            list_potentialtemperature.append(theta)
            
            n += 1

        potential_temperature = np.asarray(list_potentialtemperature)
        potential_temperature *= self.units.K
        return potential_temperature

# -----------------------------------------------------------------------------------------#

    def mixing_ratio(self, pressure, vapor_pressure):
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

        Equation:
            PW = frac{1}{rho \* g} \* \int r dp
        """
        # Defining some variables.
        cdef np.ndarray pressure = self.pressure_surfaces(dim_3d=True).to(self.units.Pa).value
        pressure = np.transpose(pressure, (2, 1, 0))
        cdef np.ndarray vapor_pressure = self.vapor_pressure().to(self.units.Pa)
        vapor_pressure = np.transpose(vapor_pressure.value, (2, 1, 0))
        cdef g = self.g
        cdef rho_w = 997 * (self.units.kg / self.units.m ** 3)

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

                P_wv = np.sum(pwv_alt)
                pwv_lat.append(P_wv)                   

                n += 1
            
            list_precipitablewater.append(pwv_lat)
            i += 1

        precipitable_water = np.asarray(list_precipitablewater) * self.units.Pa

        if np.shape(precipitable_water) == (len_pressurelong, len_pressurelat):
            precipitable_water = np.transpose(precipitable_water)
        else:
            precipitable_water = np.transpose(precipitable_water, (2, 1, 0))

        # Multiplication term by integration.
        precipitable_water *= -1 / (rho_w * g)

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
        latitude, longitude = self.latitude_lines(), self.longitude_lines()
        cdef np.ndarray precipitable_water = self.precipitable_water()

        # EckertIII projection details.
        ax = plt.axes(projection=ccrs.EckertIII())
        ax.set_global()
        ax.coastlines()
        ax.gridlines()

        # Contourf plotting.
        minimum = precipitable_water.min()
        maximum = precipitable_water.max()
        levels = np.linspace(minimum, maximum, 21)
        precipitable_water, lon = add_cyclic_point(
            precipitable_water, coord=self.longitude_lines()
        )
        plt.contourf(
            lon,
            latitude,
            precipitable_water,
            transform=ccrs.PlateCarree(),
            levels=levels,
        )

        # Add SALT.
        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Longitude ($\lambda$)")
        if not self.input_data:
            plt.title("Precipitable Water" + " ("
                + self.data_date[0] + '-' + self.data_date[1] + '-'
                + self.data_date[2] + " " + self.data_date[3] + ":00 h)"
            )
        else:
            plt.title("Precipitable Water")

        # Colorbar creation.
        colorbar = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=15)
        colorbar.locator = tick_locator
        colorbar.update_ticks()
        colorbar.set_label("Precipitable Water (mm)")

        plt.show()
        plt.close()