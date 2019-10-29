#cython: language_level=3
"""
AMSIMP Backend Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import os
from datetime import datetime
import io
import socket
import wget
import dateutil.relativedelta as future_month
import numpy as np
from astropy import constants as const
from astropy import units
from scipy.constants import gas_constant
from scipy.optimize import curve_fit
from pynverse import inversefunc
import requests
cimport numpy as np
from cpython cimport bool
import matplotlib.pyplot as plt
from matplotlib import ticker

# -----------------------------------------------------------------------------------------#


cdef class Backend:
    """
    AMSIMP Backend Class - This is the base class for AMSIMP, as such, all
    other classes within AMSIMP inherit this class, either directly or
    indirectly.

    For methods to be included in this class they must meet one of the following
    criteria:
    (1) they are considered essential components of the software.
    (2) the output of these methods are generated from the methods classified
    as (1).
    (3) these methods import data from NRLMSISE-00.
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

    temperature ~ outputs a NumPy array of temperature (3).
    density ~ outputs a NumPy array of atmospheric density (3).
    
    pressure ~ outputs a NumPy array of atmospheric pressure (4).
    pressure_thickness ~ outputs a NumPy array of atmospheric pressure
    thickness (4).
    potential_temperature ~ outputs a NumPy array of potential temperature (4).
    exner_function ~ outputs a NumPy array of the Exner function (4).
    troposphere_boundaryline ~ generates a NumPy array of the mean
    troposphere - stratosphere boundary line (4).
    temperature_contourf ~ generates a temperature contour plot (4).
    """

    # Define units of measurement for AMSIMP.
    units = units

    # Current month. the number of days in it.
    date = datetime.now()
    month = date.strftime("%B").lower()
    # Next month.
    future_date = date + future_month.relativedelta(months=+1)
    next_month = future_date.strftime("%B").lower()
    # The number of days in the current month.
    number_of_days = (future_date - date).days

    # Predefined Constants.
    # Angular rotation rate of Earth.
    sidereal_day = ((23 + (56 / 60)) * 3600) * units.s
    Upomega = ((2 * np.pi) / sidereal_day) * units.rad
    # Ideal Gas Constant
    R = gas_constant * (units.J / (units.mol * units.K))
    # Mean radius of the Earth.
    a = const.R_earth
    a = a.value * units.m
    # Universal Gravitational Constant.
    G = const.G
    G = G.value * G.unit
    # Mass of the Earth.
    M = const.M_earth
    M = M.value  * M.unit
    # Molar mass of the Earth's air.
    # m = 0.02896 (Unnecessary?)
    # The specific heat capacity on a constant pressure surface for dry air.
    c_p = 29.100609300000002 * (units.J / (units.mol * units.K))
    # Gravitational acceleration at the Earth's surface.
    g = (G * M) / (a ** 2)

    def __cinit__(self, int detail_level=3, bool future=False):
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

        # Ensure self.detail_level is between 1 and 5.
        if self.detail_level > 5 or self.detail_level <= 0:
            raise Exception(
                "detail_level must be a positive integer between 1 and 5. The value of detail_level was: {}".format(
                    self.detail_level
                )
            )

        # Check for an internet connection.
        def is_connected():
            try:
                host = socket.gethostbyname("www.github.com")
                s = socket.create_connection((host, 80), 2)
                s.close()
                return True
            except OSError:
                pass
            return False

        if not is_connected():
            raise Exception(
                "You must connect to the internet in order to utilise AMSIMP."
                + " Apologises for any inconvenience caused."
            )

    cpdef np.ndarray latitude_lines(self):
        """
        Generates a NumPy array of latitude lines.
        """
        cdef float i
        latitude_lines = [
            i
            for i in np.arange(-90, 91, (180 / (5 ** (3 - 1))))
            if -90 <= i <= 90 and i != 0
        ]
        
        # Remove the North and South Pole from the list.
        del latitude_lines[0]
        del latitude_lines[-1]

        # Convert list to NumPy array and add the unit of measurement.
        latitude_lines = np.asarray(latitude_lines) * units.deg

        # Adjust array in accordance with the level of detail specified.
        if self.detail_level == 4:
            latitude_lines = latitude_lines[::2]
        elif self.detail_level == 3:
            latitude_lines = latitude_lines[1::3]
        elif self.detail_level == 2:
            latitude_lines = latitude_lines[2::4]
        elif self.detail_level ==1:
            latitude_lines = latitude_lines[2::5]

        return latitude_lines
    
    cpdef np.ndarray longitude_lines(self):
        """
        Generates a NumPy array of longitude lines.
        """
        longitude_lines = self.latitude_lines() * 2

        return longitude_lines

    cpdef np.ndarray altitude_level(self):
        """
        Generates a NumPy array of altitude levels.
        """
        cdef int max_height = 50000
        mim_height_detail = max_height / 5

        cdef float i
        altitude_level = [
            i
            for i in np.arange(
                0, max_height + 1, (mim_height_detail / (5 ** (3 - 1)))
            )
            if i <= max_height
        ]

        altitude_level = np.asarray(altitude_level) * units.m
        return altitude_level

# -----------------------------------------------------------------------------------------#

    cpdef np.ndarray coriolis_parameter(self):
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

    cpdef np.ndarray gravitational_acceleration(self):
        """
        Generates a NumPy arrray of the effective gravitational acceleration
        according to WGS84 at a distance z from the Globe. The output is
        in the shape of (len(latitude_lines), len(altitude_level())). There
        is no longitudinal variation in gravitational accleration.
        """
        cdef np.ndarray latitude = np.radians(self.latitude_lines().value)
        cdef float a = 6378137
        cdef float b = 6356752.3142
        cdef float g_e = 9.7803253359
        cdef float g_p = 9.8321849378

        # Magnitude of the effective gravitational acceleration according to WGS84
        # at point P on the ellipsoid.
        g_0 = (
            (a * g_e * (np.cos(latitude) ** 2)) + (b * g_p * (np.sin(latitude) ** 2))
        ) / np.sqrt(
            ((a ** 2) * (np.cos(latitude) ** 2) + (b ** 2) * (np.sin(latitude) ** 2))
        )

        # Magnitude of the effective gravitational acceleration according to WGS84 at
        # a distance z from the ellipsoid.
        cdef float f = (a - b) / a
        cdef float m = (
            (self.Upomega.value ** 2) * (a ** 2) * b
            ) / (self.G.value * self.M.value)

        gravitational_acceleration = []
        cdef float z
        cdef np.ndarray altitude = self.altitude_level().value
        for z in altitude:
            g_z = g_0 * (
                1
                - (2 / a) * (1 + f + m - 2 * f * (np.sin(latitude) ** 2)) * z
                + (3 / (a ** 2)) * (z ** 2)
            )
            g_z = g_z.tolist()
            gravitational_acceleration.append(g_z)

        gravitational_acceleration = np.asarray(gravitational_acceleration)
        gravitational_acceleration = np.transpose(gravitational_acceleration)
        gravitational_acceleration *= (units.m / (units.s ** 2))
        return gravitational_acceleration

# -----------------------------------------------------------------------------------------#

    cpdef np.ndarray temperature(self):
        """
        This method imports temperature data from a NumPy file, which is
        located in the AMSIMP data repository on GitHub. The data
        stored within this repo is updated every six hours by amsimp-bot.
        Following which, it outputs a NumPy array in the shape of
        (len(longitude_lines), len(latitude_lines), len(altitude_level)).

        Temperature is defined as the mean kinetic energy density of molecular
        motion.

        I strongly recommend storing the output into a variable in order to
        prevent the need to repeatly download the file. For more information,
        visit https://github.com/amsimp/amsimp-data.
        """
        # The url to the NumPy temperature file stored on the AMSIMP data repository.
        url = "https://github.com/amsimp/amsimp-data/raw/master/temperature.npy"

        # Download the NumPy file.
        temperature_file = wget.download(url, bar=None)

        # Load and store the NumPy array into a variable.
        temperature_array = np.load(temperature_file)

        # Remove the NumPy file in order to prevent clutter.
        os.remove(temperature_file)
        
        # Adjust the temperature array in order to meet the level of detail specified.
        if self.detail_level == 5:
            temperature = temperature_array
        elif self.detail_level == 4:
            temperature = temperature_array[::2, ::2, :]
        elif self.detail_level == 3:
            temperature = temperature_array[1::3, 1::3, :]
        elif self.detail_level == 2:
            temperature = temperature_array[2::4, 2::4, :]
        elif self.detail_level == 1:
            temperature = temperature_array[2::5, 2::5, :]

        # Define the unit of measurement for temperature.
        temperature *= units.K
        return temperature
    
    cpdef np.ndarray density(self):
        """
        This method imports atmospheric density data from a NumPy file, 
        which is located in the AMSIMP data repository on GitHub. The data
        stored within this repo is updated every six hours by amsimp-bot.
        Following which, it outputs a NumPy array in the shape of
        (len(longitude_lines), len(latitude_lines), len(altitude_level)).

        The atmospheric density is the mass of the atmosphere per unit
        volume.

        I strongly recommend storing the output into a variable in order to
        prevent the need to repeatly download the file. For more information,
        visit https://github.com/amsimp/amsimp-data.
        """
        # The url to the NumPy density file stored on the AMSIMP data repository.
        url = "https://github.com/amsimp/amsimp-data/raw/master/density.npy"

        # Download the NumPy file.
        density_file = wget.download(url, bar=None)

        # Load and store the NumPy array into a variable.
        density_array = np.load(density_file)

        # Remove the NumPy file in order to prevent clutter.
        os.remove(density_file)
        
        # Adjust the density array in order to meet the level of detail specified.
        if self.detail_level == 5:
            density = density_array
        elif self.detail_level == 4:
            density = density_array[::2, ::2, :]
        elif self.detail_level == 3:
            density = density_array[1::3, 1::3, :]
        elif self.detail_level == 2:
            density = density_array[2::4, 2::4, :]
        elif self.detail_level == 1:
            density = density_array[2::5, 2::5, :]

        # Define the unit of measurement for atmospheric density.
        density *= (units.kg / (units.m ** 3))
        return density

# -----------------------------------------------------------------------------------------#

    cpdef np.ndarray pressure(self):
        """
        Description is placed here.

        Pressure is defined as the flux of momentum component normal to a given
        surface.
        """
        # Universal gas constant (J * kg^-1 * K^-1)
        R = 287 * (units.J / (units.kg * units.K))
    
        pressure = self.density() * R * self.temperature()

        pressure = pressure.to(units.hPa)

        return pressure
    
    cpdef fit_method(self, x, a, b, c):
        """
        This method is solely utilised for non-linear regression in the
        amsimp.Backend.pressure_thickness() method. Please do not
        interact with the method directly.
        """
        return a - (b / c) * (1 - np.exp(-c * x))

    cpdef np.ndarray pressure_thickness(self):
        """
        Description to be added.

        Pressure thickness is the distance between two pressure surfaces.
        """
        cdef np.ndarray pressure = self.pressure().value
        cdef np.ndarray altitude = self.altitude_level()[:20].value

        cdef np.ndarray p, p_lat, abc
        cdef list list_pressurethickness = []
        cdef list guess = [1000, 0.12, 0.00010]
        cdef list pthickness_lat
        cdef tuple c
        cdef float hPa1000_height, hPa500_height, pthickness
        for p in pressure:
            pthickness_lat =  []
            for p_lat in p:
                p_lat = p_lat[:20]

                c = curve_fit(self.fit_method, altitude, p_lat, guess)
                abc = c[0]

                inverse_fitmethod = inversefunc(self.fit_method,
                    args=(abc[0], abc[1], abc[2])
                )

                hPa1000_height = inverse_fitmethod(1000)
                hPa500_height = inverse_fitmethod(500)
                pthickness = hPa500_height - hPa1000_height

                pthickness_lat.append(pthickness)
            list_pressurethickness.append(pthickness_lat)
        
        pressure_thickness = np.asarray(list_pressurethickness)
        pressure_thickness *= units.m
        return pressure_thickness

    cpdef np.ndarray potential_temperature(self):
        """
        Generates a NumPy array of potential temperature. The potential
        temperature of a parcel of fluid at pressure P is the temperature that
        the parcel would attain if adiabatically brought to a standard reference
        pressure
        """
        cdef np.ndarray temperature = self.temperature().value
        cdef np.ndarray pressure = self.pressure().value
        cdef float R = -self.R.value
        cdef float c_p = self.c_p.value

        cdef np.ndarray array_potentialtemperature
        cdef list list_potentialtemperature = []
        cdef list p_temp
        cdef int n = 0
        cdef float len_pressure = len(pressure[0][0])
        while n < len_pressure:
            array_potentialtemperature = temperature[:, :, n] * (
                (pressure[:, :, n] / pressure[:, :, 0]) ** (R / c_p)
            )

            p_temp = list(array_potentialtemperature)

            list_potentialtemperature.append(p_temp)

            n += 1
        
        potential_temperature = np.asarray(list_potentialtemperature)
        potential_temperature = np.transpose(potential_temperature, (1, 2, 0))
        potential_temperature *= units.K
        return potential_temperature

    cpdef np.ndarray exner_function(self):
        """
        Generates a NumPy array of the exner function. The Exner function can be
        viewed as non-dimensionalized pressure.
        """
        cdef np.ndarray temperature = self.temperature()
        cdef np.ndarray potential_temperature = self.potential_temperature()

        exner_function = temperature / potential_temperature

        return exner_function

    cpdef np.ndarray troposphere_boundaryline(self):
        """
        Generates a NumPy array of the troposphere - stratosphere
        boundary line in the shape (len(longitude_lines), len(latitude_lines).
        This is calculated by looking at the vertical temperature profile in
        the method, amsimp.Backend.temperature().
        """
        cdef np.ndarray temperature = self.temperature().value
        grad_temperature = np.gradient(temperature)
        cdef np.ndarray gradient_temperature = grad_temperature[2]

        cdef np.ndarray altitude = self.altitude_level().value

        cdef list trop_strat_line = []
        cdef list lat_trop_strat_line
        cdef np.ndarray temp, t
        cdef int n
        cdef float t_n, alt
        for temp in gradient_temperature:
            lat_trop_strat_line = []
            for t in temp:
                n = 0
                while n < len(t):
                    t_n = t[n]
                    if t[n] >= 0:
                        alt = altitude[n]
                        lat_trop_strat_line.append(alt)
                        n = len(t)
                    n += 1
            trop_strat_line.append(lat_trop_strat_line)

        troposphere_boundaryline = np.asarray(trop_strat_line) * units.m
        return troposphere_boundaryline

    def temperature_contourf(self, central_long=-7.6921):
        """
        Generates a temperature contour plot, with the axes being latitude,
        and altitude. For the raw data, please use the
        amsimp.Backend.temperature() method.
        """
        # Ensure central_long is between -180 and 180.
        if central_long < -180 or central_long > 180:
            raise Exception(
                "central_long must be a real number between -180 and 180. The value of central_long was: {}".format(
                    central_long
                )
            )
        
        # Index of the nearest central_long in amsimp.Backend.longtitude_lines()
        indx_long = (np.abs(self.longitude_lines().value - central_long)).argmin()

        # Defines the axes, and the data.
        latitude, altitude = np.meshgrid(self.latitude_lines(), self.altitude_level())
        temperature = self.temperature()[indx_long, :, :]
        temperature = np.transpose(temperature)

        # Contourf plotting.
        cmap = plt.get_cmap("hot")
        minimum = temperature.min()
        maximum = temperature.max()
        levels = np.linspace(minimum, maximum, 21)
        plt.contourf(
            latitude,
            altitude,
            temperature,
            cmap=cmap,
            levels=levels,
        )

        # Adds SALT to the graph.
        if self.future:
            month = self.next_month.title()
        else:
            month = self.month.title()

        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Altitude (m)")
        plt.title("Temperature Contour Plot ("
         + self.date.strftime("%d-%m-%Y") + ", Longitude = "
         + str(np.round(self.longitude_lines()[indx_long], 2)) + ")"
        )

        # Colorbar creation.
        colorbar = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=15)
        colorbar.locator = tick_locator
        colorbar.update_ticks()
        colorbar.set_label("Temperature (K)")

        # Average boundary line between the troposphere and the stratosphere.
        troposphere_boundaryline = self.troposphere_boundaryline()
        avg_tropstratline = np.mean(troposphere_boundaryline) + np.zeros(
            len(troposphere_boundaryline[0])
        )

        # Plot average boundary line on the contour plot.
        plt.plot(
            latitude[1],
            avg_tropstratline,
            color="black",
            linestyle="dashed",
            label="Troposphere - Stratosphere Boundary Line",
        )
        plt.legend(loc=0)

        plt.show()
