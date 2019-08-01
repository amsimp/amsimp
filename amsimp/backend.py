"""
AMSIMP Backend Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
from datetime import datetime
import dateutil.relativedelta as future
import numpy as np
from astropy import constants as const
from scipy.constants import gas_constant
from scipy.optimize import curve_fit
import pandas as pd
import io
import requests

# -----------------------------------------------------------------------------------------#


class Backend:
    """
    AMSIMP Backend Class - This is the base class for AMSIMP, as such, all
    other classes within AMSIMP inherit this class, either directly or
    indirectly.

    For methods to be included in this class they must meet one of the following
    criteria:
    (1) they are considered essential components of the software.
    (2) the output of these methods are generated from the methods classified
    as (1).
    (3) these methods import data from a comma-separated values file.
    (4) they don't classify nicely into any other class.

    Below is a list of the methods included within this class, with a short
    description of their intended purpose and a bracketed number signifying
    which of the above criteria they meet. Please see the relevant class methods
    for more information.

    latitude_lines ~ generates a NumPy array of latitude lines (1).
    altitude_level ~ generates a NumPy array of altitude levels (1).

    coriolis_parameter ~ generates a NumPy arrray of the Coriolis parameter (2).
    gravitational_acceleration ~ generates a NumPy arrray of the gravitational
    acceleration (2).

    geopotential_height ~ outputs a NumPy arrray of geopotential height (3).
    temperature ~ outputs a NumPy array of temperature (3).
    pressure ~ outputs a NumPy array of atmospheric pressure (3).
    pressure_thickness ~ outputs a NumPy array of atmospheric pressure
    thickness (3).

    density ~ outputs a NumPy array of atmospheric density (4).
    potential_temperature ~ outputs a NumPy array of potential temperature (4).
    exner_function ~ outputs a NumPy array of the Exner function (4).
    troposphere_boundaryline ~ generates a NumPy array of the mean
    troposphere - stratosphere boundary line.
    """

    # Current month. the number of days in it.
    date = datetime.now()
    month = date.strftime("%B").lower()
    # Next month.
    future_date = date + future.relativedelta(months=+1)
    next_month = future_date.strftime("%B").lower()
    # The number of days in the current month.
    number_of_days = (future_date - date).days

    # Predefined Constants.
    # Angular rotation rate of Earth.
    sidereal_day = (23 + (56 / 60)) * 3600
    Upomega = (2 * np.pi) / sidereal_day
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

    def __init__(self, detail_level=3, future=False):
        """
        The parameter, detail_level, is the numerical value for the level of
        computational detail that will be used in the mathematical calculations.
        This value is between 1 and 5.

        If the parameter, future, has a boolean truth value, all class methods
        will output information related to the following month (e.g. it will
        output information related to August if the current month is July).
        """
        # Make these parameters available globally.
        self.detail_level = detail_level
        self.future = future

        # Ensure self.detail_level is an integer.
        if not isinstance(self.detail_level, int):
            raise Exception(
                "detail_level must be an integer. The value of detail_level was: {}".format(
                    self.detail_level
                )
            )

        # Ensure self.detail_level is between 1 and 5.
        if self.detail_level > 5 or self.detail_level <= 0:
            raise Exception(
                "detail_level must be a positive integer between 1 and 5. The value of detail_level was: {}".format(
                    self.detail_level
                )
            )

        self.detail_level -= 1
        self.detail_level = 5 ** self.detail_level

        # Ensure self.future is a boolean value.
        if not isinstance(self.future, bool):
            raise Exception(
                "future must be a boolean value. The value of benchmark was: {}".format(
                    self.future
                )
            )

    def latitude_lines(self):
        """
        Generates a NumPy array of latitude lines.
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
        Generates a NumPy array of altitude levels.
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

    def coriolis_parameter(self):
        """
        Generates a NumPy arrray of the Coriolis parameter at various latitudes
        of the Earth's surface. The Coriolis parameter is defined as two times
        the angular rotation of the Earth by the sin of the latitude you are
        interested in.
        """
        coriolis_parameter = (
            2 * self.Upomega * np.sin(np.radians(self.latitude_lines()))
        )

        return coriolis_parameter

    def gravitational_acceleration(self):
        """
        Generates a NumPy arrray of the effective gravitational acceleration
        according to WGS84 at a distance z from the Globe.
        """
        latitude = np.radians(self.latitude_lines())
        a = 6378137
        b = 6356752.3142
        g_e = 9.7803253359
        g_p = 9.8321849378

        # Magnitude of the effective gravitational acceleration according to WGS84
        # at point P on the ellipsoid.
        g_0 = (
            (a * g_e * (np.cos(latitude) ** 2)) + (b * g_p * (np.sin(latitude) ** 2))
        ) / np.sqrt(
            ((a ** 2) * (np.cos(latitude) ** 2) + (b ** 2) * (np.sin(latitude) ** 2))
        )

        # Magnitude of the effective gravitational acceleration according to WGS84 at
        # a distance z from the ellipsoid.
        f = (a - b) / a
        m = ((self.Upomega ** 2) * (a ** 2) * b) / (self.G * self.M)

        gravitational_acceleration = []
        for z in self.altitude_level():
            g_z = g_0 * (
                1
                - (2 / a) * (1 + f + m - 2 * f * (np.sin(latitude) ** 2)) * z
                + (3 / (a ** 2)) * (z ** 2)
            )
            g_z = g_z.tolist()
            gravitational_acceleration.append(g_z)

        gravitational_acceleration = np.asarray(gravitational_acceleration)
        return gravitational_acceleration

    # -----------------------------------------------------------------------------------------#

    def geopotential_height(self):
        """
        This method imports geopotential height data from a comma-separated
        values file, which is located within a folder labelled
        'amsimp/data/geopotential_height'. Following which, it outputs a NumPy
        array in the shape of (len(altitude_level), len(latitude_lines)).

        To define geopotential height it is better to look at an example.
        If the geopotential height of a pressure surface of '500 hPa' is
        '6000 m' at a given latitude and longitude, it says that at that given
        point the pressure at 6000 m is 500 hPa.
        """
        # Location of file.
        url = "https://raw.githubusercontent.com/amsimp/amsimp/master/"
        file_folder = url + "amsimp/data/geopotential_height/"

        # Determine which month of data to import.
        if not self.future:
            file = file_folder + self.month + ".csv"
        else:
            file = file_folder + self.next_month + ".csv"

        # Import the data as a dataframe.
        s = requests.get(file).content
        data = pd.read_csv(io.StringIO(s.decode('utf-8')))

        # Generates a NumPy array of latitude values that match the given dataset.
        column_values = np.asarray([i for i in np.arange(-80, 81, 5)])

        # Hypsometric equation.
        scale_height = data["Scale Height"].values
        pressure_surface = data["Pressure (hPa)"].values
        geometric_height = (
            np.log(pressure_surface[0] / pressure_surface) * scale_height
        ) * 1000

        # Generate a NumPy array that matches the shape of latitude_lines.
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
                    if x1 <= z < x2:
                        y = (m * z) + c
                        geo_alt.append(y)
                    elif z > x2 and x2 > 47000:
                        y = (m * z) + c
                        geo_alt.append(y)

                n += 1

            potential_height.append(geo_alt)
        potential_height = np.transpose(np.asarray(potential_height))

        # Generate the final output from the above NumPy array, potential_height.
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
                    if x1 <= phi < x2:
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
        This method imports temperature data from a comma-separated values file,
        which is located within a folder labelled 'amsimp/data/temperature'.
        Following which, it outputs a NumPy array in the shape of
        (len(altitude_level), len(latitude_lines)).

        Temperature is defined as the mean kinetic energy density of molecular
        motion.
        """
        # Location of file.
        url = "https://raw.githubusercontent.com/amsimp/amsimp/master/"
        file_folder = url + "amsimp/data/temperature/"

        # Determine which month of data to import.
        if not self.future:
            file = file_folder + self.month + ".csv"
        else:
            file = file_folder + self.next_month + ".csv"

        # Import the data as a dataframe.
        s = requests.get(file).content
        data = pd.read_csv(io.StringIO(s.decode('utf-8')))

        # Generates a NumPy array of latitude values that match the given dataset.
        column_values = np.asarray([i for i in np.arange(-80, 81, 10)])

        # Generate a NumPy array that matches the shape of latitude_lines.
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
                    if x1 <= z < x2:
                        y = (m * z) + c
                        temp_alt.append(y)
                    elif z == x2 and x2 == 50000:
                        y = (m * z) + c
                        temp_alt.append(y)

                n += 1

            temp.append(temp_alt)
        temp = np.transpose(np.asarray(temp))

        # Generate the final output from the above NumPy array, temp.
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
                    if x1 <= phi < x2:
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

    def pressure(self):
        """
        This method imports pressure data from a comma-separated values file,
        which is located within a folder labelled 'amsimp/data/pressure'.
        Following which, it outputs a NumPy array in the shape of
        (len(altitude_level), len(latitude_lines)).

        Pressure is defined as the flux of momentum component normal to a given
        surface.
        """
        # Location of file.
        url = "https://raw.githubusercontent.com/amsimp/amsimp/master/"
        file_folder = url + "amsimp/data/pressure/"

        # Determine which month of data to import.
        if not self.future:
            file = file_folder + self.month + ".csv"
        else:
            file = file_folder + self.next_month + ".csv"

        # Import the data as a dataframe.
        s = requests.get(file).content
        data = pd.read_csv(io.StringIO(s.decode('utf-8')))

        # Generates a NumPy array of latitude values that match the given dataset.
        column_values = np.asarray([i for i in np.arange(-80, 81, 10)])

        # Generate a NumPy array that matches the shape of latitude_lines.
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

        # Generate the final output from the above NumPy array, p.
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
                    if x1 <= phi < x2:
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
        This method imports geopotential height data from a comma-separated
        values file, which is located within a folder labelled
        'amsimp/data/geopotential_height'. Following which, it outputs a NumPy
        array, with pressure thickness values (1000 hPa - 500 hPa), in the shape
        of latitude_lines.

        Pressure thickness is the distance between two pressure surfaces.
        """
        # Location of file.
        url = "https://raw.githubusercontent.com/amsimp/amsimp/master/"
        file_folder = url + "amsimp/data/geopotential_height/"

        # Determine which month of data to import.
        if not self.future:
            file = file_folder + self.month + ".csv"
        else:
            file = file_folder + self.next_month + ".csv"

        # Import the data as a dataframe.
        s = requests.get(file).content
        data = pd.read_csv(io.StringIO(s.decode('utf-8')))

        # Generates a NumPy array of latitude values that match the given dataset.
        column_values = np.asarray([i for i in np.arange(-80, 81, 5)])

        # Pressure surfaces.
        mb_1000 = data.iloc[0].values[2:]
        mb_500 = data.iloc[3].values[2:]

        # Pressure thickness between 1000 hPa and 500 hPa.
        p_thickness = mb_500 - mb_1000

        # Generate a NumPy array that matches the shape of latitude_lines.
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
                if x1 <= phi < x2:
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
        return pressure_thickness

    # -----------------------------------------------------------------------------------------#

    def density(self):
        """
        Generates a NumPy array of atmospheric density by utilizing the Ideal
        Gas Law.
        """
        R = 287.05

        density = self.pressure() / (R * self.temperature())

        return density

    def potential_temperature(self):
        """
        Generates a NumPy array of potential temperature. The potential
        temperature of a parcel of fluid at pressure P is the temperature that
        the parcel would attain if adiabatically brought to a standard reference
        pressure
        """
        potential_temperature = self.temperature() * (
            (self.pressure() / self.pressure()[0]) ** (-self.R / self.c_p)
        )

        return potential_temperature

    def exner_function(self):
        """
        Generates a NumPy array of the exner function. The Exner function can be
        viewed as non-dimensionalized pressure.
        """
        exner_function = self.temperature() / self.potential_temperature()

        return exner_function

    def troposphere_boundaryline(self):
        """
        Generates a NumPy array of the mean troposphere - stratosphere
        boundary line in the shape of the output of latitude_lines. This is
        calculated by looking at the vertical temperature profile in the method,
        temperature.
        """
        temperature = np.transpose(self.temperature())

        trop_strat_line = []
        for temp in temperature:
            n = 0
            while n < (len(temp)):
                y1 = temp[n]
                y2 = temp[n + 1]

                if (y2 - y1) > 0 and self.altitude_level()[n] >= 10000:
                    alt = self.altitude_level()[n]
                    trop_strat_line.append(alt)
                    n = len(temp)

                n += 1
        trop_strat_line = np.asarray(trop_strat_line)

        troposphere_boundaryline = np.mean(trop_strat_line) + np.zeros(
            len(trop_strat_line)
        )

        return troposphere_boundaryline
