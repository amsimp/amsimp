"""
AMSIMP Backend Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import numpy as np
from astropy import constants as const
from scipy.constants import gas_constant
import pandas as pd
from datetime import datetime
import dateutil.relativedelta as future
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------------------#


class Backend:
    """
	AMSIMP Backend Class - Contains / calculates all the variables needed to utilize the
	Primitive Equations.
	"""

    # Current month.
    date = datetime.now()
    month = date.strftime("%B").lower()
    # Next month.
    future_date = date + future.relativedelta(months = +1)
    next_month = future_date.strftime("%B").lower()

    # Predefined Constants.
    # Angular rotation rate of Earth.
    Upomega = 7.2921e-05
    # Ideal Gas Constant
    R = gas_constant
    # Mean radius of the Earth.
    a = const.R_earth
    a = a.value
    # Universal Gravitational Constant.
    G = const.G
    G = G.value
    # Mass of the Earth.
    M = const.M_earth
    M = M.value
    # Molar mass of the Earth's air.
    m = 0.02896
    # The specific heat capacity on a constant pressure surface for dry air.
    c_p = 29.100609300000002
    # Gravitational acceleration at the Earth's surface.
    g = (G * M) / (a ** 2)

    def __init__(self, detail_level=3, benchmark=False, future=False):
        """
		Numerical value for the level of computational detail that will be used in the mathematical
		calculations. This value is between 1 and 5.
		"""
        self.detail_level = detail_level
        self.benchmark = benchmark
        self.future = future

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
        
        if not isinstance(self.future, bool):
            raise Exception(
                "future must be a boolean value. The value of benchmark was: {}".format(
                    self.benchmark
                )
            )

    def latitude_lines(self):
        """
        Generates a numpy array of latitude lines.
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
		Generates a numpy arrray of the Coriolis parameter at various latitudes of the Earth's
		surface. As such, it also utilizes the constant, angular rotation of the Earth.
		"""
        coriolis_force = 2 * self.Upomega * np.sin(np.radians(self.latitude_lines()))

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

    def geopotential_height(self):
        """
        Explain code here.
        """
        file_folder = "amsimp/data/geopotential_height/"
        
        if self.future == False:
            file = file_folder + self.month + ".csv"
        elif self.future == True:
            file = file_folder + self.next_month + ".csv"

        data = pd.read_csv(file)
        column_values = np.asarray([i for i in np.arange(-80, 81, 5)])

        scale_height = data["Scale Height"].values
        pressure_surface = data["Pressure (hPa)"].values
        geometric_height = (
            np.log(pressure_surface[0] / pressure_surface) * scale_height
        ) * 1000

        potential_height = []
        for i in column_values:
            i = str(i)
            alt_data = data[i].values

            geo_alt = []
            n = 0
            while n < (len(alt_data) - 1):
                y2 = alt_data[n + 1]
                y1 = alt_data[n]
                x2 = geometric_height[n + 1]
                x1 = geometric_height[n]

                m = (y2 - y1) / (x2 - x1)
                c = y1 - (m * x1)

                for z in self.altitude_level():
                    if z >= x1 and z < x2:
                        y = (m * z) + c
                        geo_alt.append(y)
                    elif z > x2 and x2 > 47000:
                        y = (m * z) + c
                        geo_alt.append(y)

                n += 1

            potential_height.append(geo_alt)
        potential_height = np.transpose(np.asarray(potential_height))

        geopotential_height = []
        n = 0
        for geo in potential_height:
            geo_lat = []
            k = 0
            while k < (len(geo) - 1):
                y2 = geo[k + 1]
                y1 = geo[k]
                x2 = column_values[k + 1]
                x1 = column_values[k]

                m = (y2 - y1) / (x2 - x1)
                c = y1 - (m * x1)

                for phi in self.latitude_lines():
                    if phi >= x1 and phi < x2:
                        y = (m * phi) + c
                        geo_lat.append(y)
                    elif phi < x1 and x1 == -80:
                        y = (m * phi) + c
                        geo_lat.append(y)
                    elif phi > x2 and x2 == 80:
                        y = (m * phi) + c
                        geo_lat.append(y)

                k += 1

            geopotential_height.append(geo_lat)

        geopotential_height = np.asarray(geopotential_height)
        return geopotential_height

    def temperature(self):
        """
		Explain code here.
		"""
        file_folder = "amsimp/data/temperature/"
        
        if self.future == False:
            file = file_folder + self.month + ".csv"
        elif self.future == True:
            file = file_folder + self.next_month + ".csv"

        data = pd.read_csv(file)
        column_values = np.asarray([i for i in np.arange(-80, 81, 10)])

        temp = []
        for i in column_values:
            i = str(i)
            alt_data = data[i].values

            temp_alt = []
            n = 0
            while n < (len(alt_data) - 1):
                y2 = alt_data[n + 1]
                y1 = alt_data[n]
                x2 = data["Alt / Lat"].values[n + 1]
                x1 = data["Alt / Lat"].values[n]

                m = (y2 - y1) / (x2 - x1)
                c = y1 - (m * x1)

                for z in self.altitude_level():
                    if z >= x1 and z < x2:
                        y = (m * z) + c
                        temp_alt.append(y)
                    elif z == x2 and x2 == 50000:
                        y = (m * z) + c
                        temp_alt.append(y)

                n += 1

            temp.append(temp_alt)
        temp = np.transpose(np.asarray(temp))

        temperature = []
        n = 0
        for t in temp:
            temp_lat = []
            k = 0
            while k < (len(t) - 1):
                y2 = t[k + 1]
                y1 = t[k]
                x2 = column_values[k + 1]
                x1 = column_values[k]

                m = (y2 - y1) / (x2 - x1)
                c = y1 - (m * x1)

                for phi in self.latitude_lines():
                    if phi >= x1 and phi < x2:
                        y = (m * phi) + c
                        temp_lat.append(y)
                    elif phi < x1 and x1 == -80:
                        y = (m * phi) + c
                        temp_lat.append(y)
                    elif phi > x2 and x2 == 80:
                        y = (m * phi) + c
                        temp_lat.append(y)

                k += 1

            temperature.append(temp_lat)

        temperature = np.asarray(temperature)
        return temperature

    # -----------------------------------------------------------------------------------------#

    def pressure(self):
        """
		Explain code here.
		"""
        file_folder = "amsimp/data/pressure/"
        
        if self.future == False:
            file = file_folder + self.month + ".csv"
        elif self.future == True:
            file = file_folder + self.next_month + ".csv"

        data = pd.read_csv(file)
        column_values = np.asarray([i for i in np.arange(-80, 81, 10)])

        p = []
        for i in column_values:
            i = str(i)
            x = data["Alt / Lat"].values
            y = data[i].values

            def fit_method(x, a, b, c):
                return a - (b / c) * (1 - np.exp(-c * x))

            guess = [1013.256, 0.1685119, 0.00016627]
            c, cov = curve_fit(fit_method, x, y, guess)

            p_alt = fit_method(self.altitude_level(), c[0], c[1], c[2])
            p.append(p_alt)

        p = np.transpose(np.asarray(p))

        pressure = []
        for _p in p:
            p_lat = []
            k = 0
            while k < (len(_p) - 1):
                y2 = _p[k + 1]
                y1 = _p[k]
                x2 = column_values[k + 1]
                x1 = column_values[k]

                m = (y2 - y1) / (x2 - x1)
                c = y1 - (m * x1)

                for phi in self.latitude_lines():
                    if phi >= x1 and phi < x2:
                        y = (m * phi) + c
                        p_lat.append(y)
                    elif phi < x1 and x1 == -80:
                        y = (m * phi) + c
                        p_lat.append(y)
                    elif phi > x2 and x2 == 80:
                        y = (m * phi) + c
                        p_lat.append(y)

                k += 1

            pressure.append(p_lat)

        pressure = np.asarray(pressure)
        pressure *= 100

        return pressure

    def pressure_thickness(self):
        """
		Explain code here.
		"""
        file_folder = "amsimp/data/geopotential_height/"
        
        if self.future == False:
            file = file_folder + self.month + ".csv"
        elif self.future == True:
            file = file_folder + self.next_month + ".csv"

        data = pd.read_csv(file)
        column_values = np.asarray([i for i in np.arange(-80, 81, 5)])

        mb_1000 = data.iloc[0].values[2:]
        mb_500 = data.iloc[3].values[2:]

        p_thickness = mb_500 - mb_1000

        pressure_thickness = []
        n = 0
        while n < (len(column_values) - 1):
            y2 = p_thickness[n + 1]
            y1 = p_thickness[n]
            x2 = column_values[n + 1]
            x1 = column_values[n]

            m = (y2 - y1) / (x2 - x1)
            c = y1 - (m * x1)

            for phi in self.latitude_lines():
                if phi >= x1 and phi < x2:
                    y = (m * phi) + c
                    pressure_thickness.append(y)
                elif phi < x1 and x1 == -80:
                    y = (m * phi) + c
                    pressure_thickness.append(y)
                elif phi > x2 and x2 == 80:
                    y = (m * phi) + c
                    pressure_thickness.append(y)

            n += 1

        pressure_thickness = np.asarray(pressure_thickness)
        return pressure_thickness, p_thickness

    # -----------------------------------------------------------------------------------------#

    def density(self):
        """
        Generates a numpy array of atmospheric density by utilizing the Ideal Gas Law. As such,
        it was generated by utilising the class methods of temperature, and pressure.
        """
        R = 287.05

        density = self.pressure() / (R * self.temperature())

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
                (pressure[i] / self.pressure()[0]) ** (self.R / self.c_p)
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