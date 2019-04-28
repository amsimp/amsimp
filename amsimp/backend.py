"""
AMSIMP Backend Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import math

import numpy as np
from astropy import constants as const
from scipy.constants import gas_constant

# -----------------------------------------------------------------------------------------#


class Backend:
    """
	AMSIMP Backend Class - Contains / calculates all the variables needed to utilize the
	Primitive Equations.
	"""

    # Predefined Constants.
    # Angular rotation rate of Earth.
    Upomega = 7.2921e-05
    # Ideal Gas Constant
    R = gas_constant
    # Mean radius of the Earth.
    a = const.R_earth
    a = a.value
    # Big G.
    G = const.G
    G = G.value
    # Mass of the Earth.
    M = const.M_earth
    M = M.value
    # The specific heat on a constant pressure surface for dry air.
    c_p = 29.086124178397213
    # Mean little g at sea level
    g = 9.80665

    def __init__(self, detail_level=3, benchmark=False):
        """
		Numerical value for the level of computational detail that will be used in the mathematical
		calculations. This value is between 1 and 5.
		"""
        self.detail_level = detail_level
        self.benchmark = benchmark

        if not isinstance(self.detail_level, int):
            raise Exception(
                "detail_level must be an integer. The value of detail_level was: {}".format(
                    self.detail_level
                )
            )

        if self.detail_level > 5 or self.detail_level <= 0:
            raise Exception(
                "detail_level must be a positive integer between 1 and 5. The value of detail_level was: {}".format(
                    self.detail_level
                )
            )

        self.detail_level -= 1
        self.detail_level = 5 ** self.detail_level

        if not isinstance(self.benchmark, bool):
            raise Exception(
                "benchmark must be a boolean value. The value of benchmark was: {}".format(
                    self.benchmark
                )
            )

    def latitude_lines(self):
        """
        Generates a numpy of latitude lines.
        """
        latitude_lines = [
            i
            for i in np.arange(-90, 91, (180 / self.detail_level))
            if -90 <= i <= 90 and i != 0
        ]

        if self.detail_level != (5 ** (1 - 1)) and self.detail_level != (5 ** (3 - 1)):
            del latitude_lines[0]
            del latitude_lines[-1]
        elif self.detail_level == (5 ** (3 - 1)):
            del latitude_lines[0]

        latitude_lines = np.asarray(latitude_lines)
        return latitude_lines

    def longitude_lines(self):
        """
        Generates a numpy of longitude lines.
        """
        longitude_lines = [
            i
            for i in np.arange(-180, 181, (360 / self.detail_level))
            if -180 <= i <= 180 and i != 0
        ]

        if self.detail_level != (5 ** (1 - 1)) and self.detail_level != (5 ** (3 - 1)):
            del longitude_lines[0]
            del longitude_lines[-1]
        elif self.detail_level == (5 ** (3 - 1)):
            del longitude_lines[0]

        longitude_lines = np.asarray(longitude_lines)
        return longitude_lines

    def altitude_level(self):
        """
        Generates a numpy which will be used in calculations relating to the altitude above
        sea level (array in metres).
        """
        max_height = 50000
        mim_height_detail = max_height / 5

        altitude_level = [
            i
            for i in np.arange(
                0, max_height + 1, (mim_height_detail / self.detail_level)
            )
            if i <= max_height
        ]

        altitude_level = np.asarray(altitude_level)
        return altitude_level

    # -----------------------------------------------------------------------------------------#

    def coriolis_force(self):
        """
		Generates a numpy of the Coriolis parameter at various latitudes on the. Earth's
		surface. As such, it also utilizes the constant, angular rotation of the Earth.
		"""
        coriolis_force = []

        for latitude in self.latitude_lines():
            latitude = 2 * self.Upomega * math.sin(math.radians(latitude))
            coriolis_force.append(latitude)

        coriolis_force = np.asarray(coriolis_force)
        return coriolis_force

    def planetary_vorticity(self):
        """
        The planetary vorticity gradient is defined as: df/dy (the partial derivative
        of the Coriolis force with respect to latitude). This generates a numpy array
        of this.
        """
        planetary_vorticity = (
            2 * self.Upomega * np.cos(self.coriolis_force())
        ) / self.a

        return planetary_vorticity

    def geopotential(self):
        """
		Geopotential is the potential of the Earth's gravity field.
		"""
        # Defining some of variables.
        latitude = np.radians(self.latitude_lines())
        a = 6378137
        b = 6356752.3142
        epsilon = np.sqrt((a ** 2) - (b ** 2)) / a
        N_lat = a / np.sqrt(1 - epsilon * np.sin(latitude) ** 2)

        # Distance of the point P measured from the earth axis.
        R = []

        for z in self.altitude_level():
            var = (N_lat + z) * np.cos(latitude)
            var = var.tolist()
            R.append(var)

        R = np.asarray(R)

        # Magnitude of the Apparent Gravitational Acceleration
        g_A = []
        for var in R:
            x = self.g - (var * (self.Upomega ** 2) * np.cos(latitude) ** 2)
            x = x.tolist()
            g_A.append(x)
        g_A = np.asarray(g_A)

        # Calculation of geopotential
        geopotential = []
        count = 0
        for g in g_A:
            z = self.altitude_level()
            z = z.tolist()
            potential = z[count] * g
            potential = potential.tolist()
            geopotential.append(potential)
            count += 1

        geopotential = np.asarray(geopotential)
        return geopotential

    def geopotential_height(self):
        """
        Geopotential height or geopotential altitude is a vertical coordinate
        referenced to Earth's mean sea level, an adjustment to geometric height
        (altitude above mean sea level) using the variation of gravity with latitude
        and vertical position. Thus, it can be considered a "gravity-adjusted height".
        """
        geopotential_height = self.geopotential() / self.g

        return geopotential_height

    def pressure(self):
        """
		Generates a numpy of atmospheric pressure by utilizing the Barometric Formula
		for Pressure. As such, it was generated from an altitude above sea level numpy.
		Such a numpy was created in the function, altitude_level().
		"""
        pressure = []

        for altitude in self.altitude_level():
            altitude /= 1000
            altitude = -832.6777 + (101323.6 + 832.6777) / (
                1 + (altitude / 6.527821) ** 2.313703
            )
            pressure.append(altitude)

        pressure = np.asarray(pressure)
        return pressure

    # -----------------------------------------------------------------------------------------#

    def vertical_velocity(self):
        """
		Generates a numpy of vertical velocity (omega) by utilizing the derivative of
		the pressure equation (pressure() function).

		Since pressure decreases upward, a negative omega means rising motion, while
		a positive omega means subsiding motion.
		"""
        vertical_velocity = []

        for pressure in self.pressure():
            pressure = -832.6777 + 102156.2777 / (
                1.49196723444642e-9 * pressure ** 2.313703 + 1
            )
            vertical_velocity.append(pressure)

        vertical_velocity = np.asarray(vertical_velocity)
        return vertical_velocity

    def temperature(self):
        """
		These calculations are based on the International Standard Atmosphere, as such,
		the temperatures in this model only vary by height, and not by any other variable.
		I plan, however, of doing this in the future. It would require an extraordinary
		amount of coding.

		The International Standard Atmosphere is a static atmospheric model of how the pressure,
		temperature, density, and viscosity of the Earth's atmosphere change over a wide range
		of altitudes.
		"""
        temperature = []

        T_b = 288.15

        for altitude in self.altitude_level():
            # Troposphere
            if altitude <= 11000:
                altitude = T_b - (altitude * 0.0065)
                temperature.append(altitude)
            # Tropopause
            elif altitude <= 20000:
                altitude = 216.65
                temperature.append(altitude)
            # Stratosphere
            elif altitude <= 32000:
                altitude = 216.65 + ((altitude - 20000) * 0.001)
                temperature.append(altitude)
            elif altitude <= 47000:
                altitude = 228.65 + ((altitude - 32000) * 0.0028)
                temperature.append(altitude)
            # Stratopause
            else:
                altitude = 270.65
                temperature.append(altitude)

        temperature = np.asarray(temperature)
        return temperature

    # -----------------------------------------------------------------------------------------#

    def potential_temperature(self):
        """
		The potential temperature of a parcel of fluid at pressure P is the temperature
		that the parcel would attain if adiabatically brought to a standard reference
		pressure
		"""
        potential_temperature = self.temperature() * (
            (self.pressure() / self.pressure()[0]) ** (-self.R / self.c_p)
        )

        potential_temperature = np.asarray(potential_temperature)
        return potential_temperature

    def exner_function(self):
        """
		The Exner function can be viewed as non-dimensionalized pressure.
		"""
        exner_function = self.temperature() / self.potential_temperature()

        return exner_function

    def sigma(self):
        """
		A vertical coordinate for atmospheric models defined as the difference in pressure.
		"""
        sigma = (self.pressure() - self.pressure()[0]) / (
            self.pressure()[0] - self.pressure()[-1]
        )

        return sigma
