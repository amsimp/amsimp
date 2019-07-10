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
    # Molar mass of the Earth's air.
    m = 0.02896
    # The specific heat on a constant pressure surface for dry air.
    c_p = 29.086124178397213
    # Mean little g at sea level.
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
                1, max_height + 1, (mim_height_detail / self.detail_level)
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

    def gravitational_acceleration(self):
        """
        This class calculates the magintude of the effective gravitational acceleration
        according to WGS84 at a distance z from the Globe.
        """
        latitude = np.radians(self.latitude_lines())
        a = 6378137
        b = 6356752.3142
        g_e = 9.7803253359
        g_p = 9.8321849378

        """
        Magnitude of the effective gravitational acceleration according to WGS84
        at point P on the ellipsoid.
        """
        g_0 = (
            (a * g_e * (np.cos(latitude) ** 2)) + (b * g_p * (np.sin(latitude) ** 2))
        ) / np.sqrt(
            ((a ** 2) * (np.cos(latitude) ** 2) + (b ** 2) * (np.sin(latitude) ** 2))
        )

        """
        Magnitude of the effective gravitational acceleration according to WGS84 at
        a distance z from the ellipsoid.
        """
        f = (a - b) / a
        m = ((self.Upomega ** 2) * (a ** 2) * b) / (self.G * self.M)

        g_z = []
        for z in self.altitude_level():
            var = g_0 * (
                1
                - (2 / a) * (1 + f + m - 2 * f * (np.sin(latitude) ** 2)) * z
                + (3 / (a ** 2)) * (z ** 2)
            )
            var = var.tolist()
            g_z.append(var)

        g_z = np.asarray(g_z)
        return g_z

    def geopotential(self):
        """
		Geopotential is the potential of the Earth's gravity field.
		"""
        geopotential = []
        count = 0
        for g in self.gravitational_acceleration():
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

    def temperature(self):
        """
		These calculations are based on the International Standard Atmosphere. The
        International Standard Atmosphere is a static atmospheric model of how the pressure,
		temperature, density, and viscosity of the Earth's atmosphere change over a wide range
		of altitudes.
		"""
        k = 1.0 * 10 ** 200
        altitude = self.altitude_level()

        term1 = ((-0.0065 * altitude) + 288.15) / (1 + np.exp(-2 * k * altitude))
        term2 = ((0.0065 * altitude) - 71.5) / (1 + np.exp(-2 * k * (altitude - 11000)))
        term3 = ((0.001 * altitude) - 20) / (1 + np.exp(-2 * k * (altitude - 20000)))
        term4 = ((0.0018 * altitude) - 57.6) / (1 + np.exp(-2 * k * (altitude - 32000)))
        term5 = ((-0.0028 * altitude) + 131.6) / (1 + np.exp(-2 * k * (altitude - 47000)))
        term6 = ((-0.0028 * altitude) + 142.8) / (1 + np.exp(-2 * k * (altitude - 51000)))
        term7 = ((-0.0028 * altitude) + 413.45) / (1 + np.exp(-2 * k * (altitude - 71000)))

        temperature = term1 + term2 + term3 + term4 + term5 + term6 - term7

        return temperature

    # -----------------------------------------------------------------------------------------#

    def pressure(self):
        """
		Generates a numpy of atmospheric pressure by utilizing the Barometric Formula
		for Pressure. As such, it was generated from an altitude above sea level numpy.
		Such a numpy was created in the function, altitude_level().
		"""
        pressure = []

        p_0 = 101325
        g = self.gravitational_acceleration()
        z = self.altitude_level()
        T = self.temperature()

        i = 0
        while i < len(self.altitude_level()):
            var = p_0 * np.exp(-((self.m * g[i]) / (self.R * T[i])) * z[i])
            var = var.tolist()

            pressure.append(var)

            i += 1

        pressure = np.asarray(pressure)
        return pressure

    # -----------------------------------------------------------------------------------------#

    def density(self):
        """
        Generates a numpy of atmospheric density by utilizing temperature, pressure,
        and the gas constant. As such, it was generated from the class methods of
        temperature, and pressure.
        """
        density = []

        R = 287.05
        pressure = self.pressure()
        temperature = self.temperature()

        i = 0
        while i < len(temperature):
            var = pressure[i] / (R * temperature[i])

            density.append(var)

            i += 1

        density = np.asarray(density)
        return density

    def potential_temperature(self):
        """
        The potential temperature of a parcel of fluid at pressure P is the temperature
        that the parcel would attain if adiabatically brought to a standard reference
        pressure
        """
        potential_temperature = []

        pressure = self.pressure()
        temperature = self.temperature()

        i = 0
        while i < len(temperature):
            var = temperature[i] * (
                (pressure[i] / self.pressure()[0]) ** (-self.R / self.c_p)
            )

            potential_temperature.append(var)

            i += 1

        potential_temperature = np.asarray(potential_temperature)
        return potential_temperature

    def exner_function(self):
        """
		The Exner function can be viewed as non-dimensionalized pressure.
		"""
        exner_function = self.temperature() / self.potential_temperature()

        return exner_function
