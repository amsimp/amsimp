#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
# cython: embedsignature=True, binding=True
"""
Copyright (C) 2020 AMSIMP

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.
"""

# ------------------------------------------------------------------------------#

# Importing Dependencies
import os, sys, socket
from datetime import datetime
from amsimp.download cimport Download
from amsimp.download import Download
import numpy as np
from astropy import constants as constant
from astropy import units
from astropy.units.quantity import Quantity
cimport numpy as np
from cpython cimport bool
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.coord_systems import GeogCS

# ------------------------------------------------------------------------------#

cdef class Backend(Download):
    """
    This is the base class for AMSIMP, as such, all other classes within AMSIMP
    inherit this class, either directly or indirectly.
    """

    def __cinit__(
            self,
            int delta_latitude=5,
            int delta_longitude=5,
            forecast_length=72, 
            delta_t=2, 
            bool ai=True, 
            data_size=150, 
            epochs=200, 
            input_date=None, 
            bool input_data=False,
            psurfaces=None,
            lat=None,
            lon=None,
            height=None, 
            temp=None, 
            rh=None, 
            u=None, 
            v=None,
            dict constants={
                "sidereal_day": (23 + (56 / 60)) * 3600,
                "angular_rotation_rate": ((2 * np.pi) / ((23 + (56 / 60)) * 3600)),
                "planet_radius": constant.R_earth.value,
                "planet_mass": constant.M_earth.value,
                "specific_heat_capacity_psurface": 718,
                "gravitational_acceleration": 9.80665,
                "planet": "Earth"
            }
        ):
        """
        The parameters, delta_latitude and delta_longitude, defines the
        horizontal resolution between grid points within the software. The
        parameter values must be between 1 and 10 degrees if you are utilising
        the default initial conditions included within the software. Defaults
        to a value of 5 degrees.

        The parameter, input_date, allows the end-user to specify the initial
        date at which the forecast is generated from. Please note that
        currently you cannot specify dates before Janurary 2020. Please also
        note that the latest data available to the software is typically
        from three days ago. This is due to the rate at which the NOAA
        updates the data on the Global Data Assimilation System website. This
        feature is intended for situations where the default initial conditions
        included within the software are being used.

        The parameter, input_data, is a boolean value, and when set to True,
        it will allow the end-user to provide their own initial conditions
        to the software.

        The parameter, constants, is a collection of meteorologically 
        significant constants used by the software. These constants
        may be defined by the end-user, however, certain functionality
        within the software may be not work correctly.
        """
        # Make the aforementioned variables available else where in the class.
        self.delta_latitude = delta_latitude
        self.delta_longitude = delta_longitude
        self.input_date = input_date
        self.input_data = input_data

        # The date at which the initial conditions was gathered (i.e. how
        # recent the data is).
        if self.input_date == None:
            data_date_file = self.download(
                "https://github.com/amsimp/initial-conditions/raw/master/date.npy", 
                bar=False,
            )
            data_date = np.load(data_date_file)
            os.remove(data_date_file)
            date = datetime(
                int(data_date[2]),
                int(data_date[1]),
                int(data_date[0]),
                int(data_date[3]),
            )
            self.date = date
        else:        
            # Ensure input_date is of the type "datetime.datetime".
            if type(self.input_date) != datetime:
                raise Exception(
                    "input_date must of the type 'datetime.datetime'."
                    + " The value of input_date was: {}".format(
                        self.input_date
                    )
                )
            date = self.input_date
            self.date = date

        # Ensure input_data is a boolean value.
        if not isinstance(self.input_data, bool):
            raise Exception(
                "input_data must be a boolean value."
                + " The value of input_data was: {}".format(
                    self.input_data
                )
            )

        # Function to ensure that the input data is 3 dimensional.
        def dimension(input_variable):
            if np.ndim(input_variable) != 3:
                raise Exception(
                    "All input data variables (height, rh, temp, u, v)"
                    + " must be a 3 dimensional."
                )

        # Function to check if a function is a NumPy array, or a list.
        def type_check(input_variable):
            if type(input_variable) == Quantity:
                pass
            elif type(input_variable) == np.ndarray:
                pass
            elif type(input_variable) == list:
                pass
            else:
                raise Exception(
                    "All input data variables must be either NumPy arrays,"
                    + " or lists."
                )
        
        # Function to check if grid input variables are 1 dimensional.
        def dimension_grid(input_variable):
            if np.ndim(input_variable) != 1:
                raise Exception(
                    "All grid input variables (psurfaces, lat, lon) must be" 
                    + " a 1 dimensional."
                )

        # Function to convert input lists into NumPy arrays.
        def list_to_numpy(input_variable):
            if type(input_variable) == list:
                input_variable = np.asarray(input_variable)
            
            return input_variable

        if self.input_data:
            # Ensure input data variables are either NumPy arrays, or lists.
            type_check(psurfaces)
            type_check(lat)
            type_check(lon)
            type_check(height)
            type_check(rh)
            type_check(temp)
            type_check(u)
            type_check(v)

            # Check if grid input variables are 1 dimensional.
            dimension_grid(psurfaces)
            dimension_grid(lat)
            dimension_grid(lon)

            # Check if input data is 3 dimensional.
            dimension(height)
            dimension(rh)
            dimension(temp)
            dimension(u)
            dimension(v)

            # Convert input data lists to NumPy arrays.
            list_to_numpy(psurfaces)
            list_to_numpy(lat)
            list_to_numpy(lon)
            list_to_numpy(height)
            list_to_numpy(rh)
            list_to_numpy(temp)
            list_to_numpy(u)
            list_to_numpy(v)

            # Add units to input variables.
            # psurfaces variable.
            if type(psurfaces) != Quantity:
                psurfaces = psurfaces * units.hPa
            # lat variable.
            if type(lat) != Quantity:
                lat = lat * units.deg
            # lon variable.
            if type(height) != Quantity:
                lon = lon * units.deg
            # height variable.
            if type(height) != Quantity:
                height = height * units.m
            # rh variable.
            if type(rh) != Quantity:
                rh = rh * units.percent
            # temp variable.
            if type(temp) != Quantity:
               temp = temp * units.K
            # u variable.
            if type(u) != Quantity:
               u = u * (units.m / units.s)
            # v variable.
            if type(v) != Quantity:
               v = v * (units.m / units.s)

        # Ensure self.delta_latitude is between 1 and 10 degrees.
        if not input_data:
            if self.delta_latitude > 10 or self.delta_latitude < 1:
                raise Exception(
                    "delta_latitude must be a positive integer between 1 and 10."
                    + " The value of delta_latitude was: {}".format(
                        self.delta_latitude
                    )
                )

            # Ensure self.delta_longitude is between 1 and 10 degrees.
            if self.delta_longitude > 10 or self.delta_longitude < 1:
                raise Exception(
                    "delta_longitude must be a positive integer between 1 and 10."
                    + " The value of delta_longitude was: {}".format(
                        self.delta_longitude
                    )
                )

            folder = "https://github.com/amsimp/initial-conditions/raw/master/"
            folder += "initial_conditions/"

            # Define date.
            year = self.date.year
            month = self.date.month
            day = self.date.day
            hour = self.date.hour

            # Adds zero before single digit numbers.
            if day < 10:
                day = "0" + str(day)

            if month < 10:
                month =  "0" + str(month)

            if hour < 10:
                hour = "0" + str(hour)

            # Converts integers to strings.
            day = str(day)
            month = str(month)
            year = str(year)
            hour = str(hour)

            folder += (
                year + "/" + month + "/" + day + "/" + hour + "/"
            )
            # The url to the NumPy pressure surfaces file stored on the AMSIMP
            # Initial Conditions Data repository.
            url = folder + "initial_conditions.nc"

            # Download the NumPy file and store the NumPy array into a variable.
            try:
                cube = iris.load("initial_conditions.nc")
            except OSError:  
                fname = self.download(url)
                cube = iris.load(fname)
            
            height = cube.extract("geopotential_height")[0]
            temp = cube.extract("air_temperature")[0]
            rh = cube.extract("relative_humidity")[0]
            u = cube.extract("x_wind")[0]
            v = cube.extract("y_wind")[0]

        # Constants dictionary.
        # Sidereal day.
        if type(constants["sidereal_day"]) != Quantity:
            constants["sidereal_day"] = constants["sidereal_day"] * units.s
        # Angular rotation rate.
        if type(constants["angular_rotation_rate"]) != Quantity:
            constants["angular_rotation_rate"] = constants[
                "angular_rotation_rate"
            ] * (
                units.rad / units.s
            )
        # Planet radius.
        if type(constants["planet_radius"]) != Quantity:
            constants["planet_radius"] = constants["planet_radius"] * units.m
        # Planet mass.
        if type(constants["planet_mass"]) != Quantity:
            constants["planet_mass"] = constants["planet_mass"] * units.kg
        # Specific heat capacity on a constant pressure surface.
        if type(constants["specific_heat_capacity_psurface"]) != Quantity:
            constants["specific_heat_capacity_psurface"] = constants[
                "specific_heat_capacity_psurface"
            ] * (units.J / (units.kg * units.K))
        # Gravitational acceleration.
        if type(constants["gravitational_acceleration"]) != Quantity: 
            constants["gravitational_acceleration"] = constants[
                "gravitational_acceleration"
            ] * (
                units.m / (units.s ** 2)
            )
        # Planet
        planet = constants["planet"].lower()
        planet = planet.replace(" ", "")
        planet = planet.capitalize()
        # Gas constant
        R = 287.05799596 * (units.J / (units.kg * units.K))
        constants["gas_constant"] = R
        # Universal gravitational constant.
        G = constant.G
        G = G.value * G.unit
        constants["universal_gravitational_constant"] = G

        # Make the input data variables available else where in the class.
        self.psurfaces = psurfaces
        self.lat = lat
        self.lon = lon
        self.input_height = height
        self.input_rh = rh
        self.input_temp = temp
        self.input_u = u
        self.input_v = v
        self.constants = constants

        # Predefined Constants.
        # Angular rotation rate of the planet.
        self.sidereal_day = constants["sidereal_day"]
        self.Upomega = constants["angular_rotation_rate"]
        # Mean radius of the planet.
        self.a = self.constants["planet_radius"]
        # Mass of the planet.
        self.M = self.constants["planet_mass"]
        # The specific heat capacity on a constant pressure surface.
        self.c_p = self.constants["specific_heat_capacity_psurface"]
        # Gravitational acceleration.
        self.g = self.constants["gravitational_acceleration"]
        # Planet
        self.planet = self.constants["planet"]
        # Gas Constant.
        self.R = constants["gas_constant"]
        # Universal gravitational constant.
        self.G = constants["universal_gravitational_constant"]

        # Function to check for an internet connection.
        def is_connected():
            try:
                host = socket.gethostbyname("www.github.com")
                s = socket.create_connection((host, 80), 2)
                s.close()
                return True
            except OSError:
                pass
            return False

        # Check for an internet connection.
        if not input_data:
            if not is_connected():
                raise Exception(
                    "You must connect to the internet in order to utilise AMSIMP."
                    + " Apologies for any inconvenience caused."
                )

        # RNN.
        self.epochs = epochs
        self.data_size = data_size

        # Ensure epochs is an integer value.
        if not isinstance(self.epochs, int):
            raise Exception(
                "epochs must be a integer value."
                + " The value of epochs was: {}".format(
                    self.ai
                )
            )

        # Ensure epochs is a natural number.
        if not self.epochs > 0:
            raise Exception(
                "epochs must be a integer value."
                + " The value of epochs was: {}".format(
                    self.ai
                )
            )

        # Ensure data_size is an integer value.
        if not isinstance(self.data_size, int):
            raise Exception(
                "data_size must be a integer value."
                + " The value of data_size was: {}".format(
                    self.ai
                )
            )

        # Ensure data_size is a natural number and is greater than 14.
        if not self.data_size > 14:
            raise Exception(
                "data_size must be a integer value."
                + " The value of data_size was: {}".format(
                    self.ai
                )
            )

# ------------------------------------------------------------------------------#

    cpdef np.ndarray latitude_lines(self):
        r"""Generates an array of latitude lines.

        Returns
        -------
        `astropy.units.quantity.Quantity`
            Latitude lines

        See Also
        --------
        longitude_lines, pressure_surfaces
        """
        cdef float i
        if not self.input_data:
            latitude_lines = [
                i
                for i in np.arange(
                    -89, 90, self.delta_latitude
                )
            ]

            # Convert list to NumPy array and add the unit of measurement.
            latitude_lines = np.asarray(latitude_lines) * units.deg
        else:
            latitude_lines = self.lat

        return latitude_lines
    
    cpdef np.ndarray longitude_lines(self):
        r"""Generates an array of longitude lines.

        Returns
        -------
        `astropy.units.quantity.Quantity`
            Longitude lines

        See Also
        --------
        latitude_lines, pressure_surfaces
        """
        cdef float i
        if not self.input_data:
            longitude_lines = [
                i
                for i in np.arange(
                    0, 360, self.delta_longitude
                )
            ]

            # Convert list to NumPy array and add the unit of measurement.
            longitude_lines = np.asarray(longitude_lines) * units.deg
        else:
            longitude_lines = self.lon

        return longitude_lines

    cpdef np.ndarray pressure_surfaces(self, dim_3d=False):
        r"""Generates an array of constant pressure surfaces. 

        Parameters
        ----------
        dim_3d: `bool`
            If true, this will generate a 3-dimensional array
        
        Returns
        -------
        `astropy.units.quantity.Quantity`
            Constant pressure surfaces

        Notes
        -----
        This is the isobaric coordinate system.

        See Also
        --------
        latitude_lines, longitude_lines
        """
        if not self.input_data:
            pressure_surfaces = self.input_rh.coords('pressure')[0].points
            pressure_surfaces = pressure_surfaces[::-1] * units.Pa
            pressure_surfaces = pressure_surfaces.to(units.hPa)
            pressure_surfaces = pressure_surfaces.value
        else:
            pressure_surfaces = self.psurfaces.value

        # Convert Pressure Array into 3D Array.
        if dim_3d:
            list_pressure = []
            for p in pressure_surfaces:
                p = np.zeros((
                    len(self.latitude_lines()), len(self.longitude_lines())
                )) + p
                
                p = p.tolist()
                list_pressure.append(p)
            pressure_surfaces = np.asarray(list_pressure)

        pressure_surfaces *= units.hPa
        return pressure_surfaces

# ------------------------------------------------------------------------------#

    cpdef np.ndarray gradient_longitude(self, parameter=None):
        r"""Calculate the gradient of a grid of values with respect to longitude.

        Parameters
        ----------
        parameter: `astropy.units.quantity.Quantity`
            Array of values of which to calculate the gradient

        Returns
        -------
        `astropy.units.quantity.Quantity`
            The gradient calculated with respect to longitude of the original array

        Notes
        -----
        The gradient is computed using second order accurate central differences
        in the interior points. The boundaries are joined together for the purposes
        of this compuation, allowing for the use of second order accurate central 
        differences at the boundaries.

        See Also
        --------
        gradient_latitude, gradient_pressure
        """
        # Determine gradient with respect to longitude.
        cdef np.ndarray longitude = np.radians(self.longitude_lines().value)

        # Define variable types.
        cdef np.ndarray parameter_central, parameter_boundaries
        cdef np.ndarray parameter_lgmt, parameter_rgmt
        cdef np.ndarray lon_lgmt, lon_rgmt, lon_boundaries

        # Compute gradient.
        # Central points.
        parameter_central = np.gradient(
            parameter, longitude, axis=2
        )

        # Boundaries.
        # Define atmospheric parameter boundaries.
        parameter_lgmt = parameter[:, :, -3:].value
        parameter_rgmt = parameter[:, :, :3].value
        # Define longitude boundaries.
        lon_lgmt = longitude[-3:]
        lon_rgmt = longitude[:3]
        # Concatenate atmospheric parameter boundaries.
        parameter_boundaries = np.concatenate(
            (parameter_lgmt, parameter_rgmt), axis=2
        ) * parameter.unit
        # Concatenate longitude boundaries.
        lon_boundaries = np.concatenate((lon_lgmt, lon_rgmt))
        # Compute gradient at boundaries.
        parameter_boundaries = np.gradient(
            parameter_boundaries, lon_boundaries, axis=2
        )
        # Redefine parameter as computed gradient.
        parameter = parameter_central
        parameter[:, :, -1] = parameter_boundaries[:, :, 2]
        parameter[:, :, 0] = parameter_boundaries[:, :, 3]

        # Make 1d latitude array into a 3d latitude array.
        cdef np.ndarray latitude = np.radians(self.latitude_lines())
        latitude = self.make_3dimensional_array(
            parameter=latitude, axis=1
        )

        # Handle longitudinal variation with respect to latitude.
        gradient_longitude = (
            (1 / (self.a * np.cos(latitude.value))) * parameter
        )

        return gradient_longitude

    cpdef np.ndarray gradient_latitude(self, parameter=None):
        r"""Calculate the gradient of a grid of values with respect to latitude.

        Parameters
        ----------
        parameter: `astropy.units.quantity.Quantity`
            Array of values of which to calculate the gradient

        Returns
        -------
        `astropy.units.quantity.Quantity`
            The gradient calculated with respect to latitude of the original array

        Notes
        -----
        The gradient is computed using second order accurate central differences
        in the interior points and either first or second order accurate 
        one-sides (forward or backwards) differences at the boundaries.
        The returned gradient hence has the same shape as the input array.

        See Also
        --------
        gradient_longitude, gradient_pressure
        """
        # Define variable.
        cdef np.ndarray latitude = np.radians(self.latitude_lines().value)

        # Determine gradient with respect to latitude.
        gradient_latitude = np.gradient(
            parameter.value, self.a.value * latitude, axis=1
        ) * (parameter.unit / units.m)

        return gradient_latitude
    
    cpdef np.ndarray gradient_pressure(self, parameter=None):
        r"""Calculate the gradient of a grid of values with respect to pressure.

        Parameters
        ----------
        parameter: `astropy.units.quantity.Quantity`
            Array of values of which to calculate the gradient

        Returns
        -------
        `astropy.units.quantity.Quantity`
            The gradient calculated with respect to pressure of the original array

        Notes
        -----
        The gradient is computed using second order accurate central differences
        in the interior points and either first or second order accurate 
        one-sides (forward or backwards) differences at the boundaries.
        The returned gradient hence has the same shape as the input array.

        See Also
        --------
        gradient_longitude, gradient_latitude
        """
        # Define variable.
        cdef np.ndarray pressure = self.pressure_surfaces()

         # Determine gradient with respect to pressure.
        gradient_pressure = np.gradient(
            parameter.value, pressure.value, axis=0
        ) * (parameter.unit / units.hPa)

        return gradient_pressure

    cpdef np.ndarray make_3dimensional_array(self, parameter=None, axis=1):
        r"""Convert 1-dimensional array to array with three dimensions.

        Parameters
        ----------
        parameter: `astropy.units.quantity.Quantity`
            Input array to be converted 
        axis: `int`
            Defines the variable over which the given parameter varies

        Returns
        -------
        `astropy.units.quantity.Quantity`
            3-dimensional version of input array
        
        Notes
        -----
        If axis is set to 0, this variable is constant pressure surfaces. 
        If axis is set to 1, this variable is latitude.  If axis is set to
        2, this variable is longitude.
        """
        if axis == 0:
            a = self.latitude_lines().value
            b = self.longitude_lines().value
        elif axis == 1:
            a = self.pressure_surfaces().value
            b = self.longitude_lines().value
        elif axis == 2:
            a = self.pressure_surfaces().value
            b = self.latitude_lines().value

        list_parameter = []
        for param in parameter:
            unit = param.unit
            param = param.value + np.zeros((len(a), len(b)))
            param = param.tolist()
            list_parameter.append(param)
        parameter = np.asarray(list_parameter) * unit

        if axis == 1:
            parameter = np.transpose(parameter, (1, 0, 2))
        elif axis == 2:
            parameter = np.transpose(parameter, (1, 2, 0))
        
        return parameter

    cpdef dict retrieve_constants(self):
        r"""Retrieves the collection of meteorologically significant constants.

        Returns
        -------
        `dict`
            Collection of meteorologically significant constants 
        
        Notes
        -----
        These constants may be defined by the user via the
        parameter, constants, on the initialisation of the class.
        """
        return self.constants

# ------------------------------------------------------------------------------#

    cpdef np.ndarray coriolis_parameter(self):
        r"""Generates an arrray of the Coriolis parameter at various latitudes.

        .. math:: f = 2 \Omega \sin{\phi}

        Returns
        -------
        `astropy.units.quantity.Quantity`
            Coriolis parameter

        Notes
        -----      
        The Coriolis parameter is defined as two times the angular rotation of
        the Earth by the sin of the latitude you are interested in.
        """
        coriolis_parameter = (
            2 * self.Upomega * np.sin(np.radians(self.latitude_lines()))
        )

        return coriolis_parameter

# ------------------------------------------------------------------------------#

    cpdef np.ndarray geopotential_height(self):
        r"""Generates an arrray of geopotential height.
        
        Returns
        -------
        `astropy.units.quantity.Quantity`
            Geopotential height

        Notes
        -----
        If the user did not define initial conditions on initialisation
        of the class, this data is retrieved from the AMSIMP Initial
        Conditions Data Repository on GitHub. Geopotential Height is the
        height above sea level of a pressure level. For example, if a 
        station reports that the 500 hPa height at its location is 5600 m,
        it means that the level of the atmosphere over that station at which
        the atmospheric pressure is 500 hPa is 5600 meters above sea level.

        See Also
        --------
        relative_humidity, temperature
        """
        if not self.input_data:
            # Input data.
            height = self.input_height

            pressure = self.pressure_surfaces().to(units.Pa)
            # Grid.
            grid_points = [
                ('pressure',  pressure.value),
                ('latitude',  self.latitude_lines().value),
                ('longitude', self.longitude_lines().value),                
            ]

            # Interpolation
            height = height.interpolate(
                grid_points, iris.analysis.Linear()
            )

            # Get data.
            geopotential_height = height.data
            geopotential_height = np.asarray(geopotential_height.tolist())

            geopotential_height *= units.m
        else:
            geopotential_height = self.input_height
        
        return geopotential_height

    cpdef np.ndarray relative_humidity(self):
        r"""Generates an arrray of relative humidity.
        
        Returns
        -------
        `astropy.units.quantity.Quantity`
            Relative humidity

        Notes
        -----
        If the user did not define initial conditions on initialisation
        of the class, this data is retrieved from the AMSIMP Initial
        Conditions Data Repository on GitHub. Relative Humidity is the amount
        of water vapour present in air expressed as a percentage of the amount
        needed for saturation at the same temperature.

        See Also
        --------
        geopotential_height, temperature
        """
        if not self.input_data:
            # Input data.
            rh = self.input_rh

            pressure = self.pressure_surfaces().to(units.Pa)
            # Grid.
            grid_points = [
                ('pressure',  pressure.value),
                ('latitude',  self.latitude_lines().value),
                ('longitude', self.longitude_lines().value),                
            ]

            # Interpolation
            rh = rh.interpolate(
                grid_points, iris.analysis.Linear()
            )

            # Get data.
            relative_humidity = rh.data
            relative_humidity = np.asarray(relative_humidity.tolist())

            relative_humidity *= units.percent
        else:
            relative_humidity = self.input_rh
        
        return relative_humidity

    cpdef np.ndarray temperature(self):
        r"""Generates an arrray of temperature.
        
        Returns
        -------
        `astropy.units.quantity.Quantity`
            Temperature

        Notes
        -----
        If the user did not define initial conditions on initialisation
        of the class, this data is retrieved from the AMSIMP Initial
        Conditions Data Repository on GitHub. Temperature is defined as
        the mean kinetic energy density of molecular motion.

        See Also
        --------
        geopotential_height, relative_humidity
        """
        if not self.input_data:
            # Input data.
            temp = self.input_temp

            pressure = self.pressure_surfaces().to(units.Pa)
            # Grid.
            grid_points = [
                ('pressure',  pressure.value),
                ('latitude',  self.latitude_lines().value),
                ('longitude', self.longitude_lines().value),                
            ]

            # Interpolation
            temp = temp.interpolate(
                grid_points, iris.analysis.Linear()
            )

            # Get data.
            temperature = temp.data
            temperature = np.asarray(temperature.tolist())

            temperature *= units.K
        else:
            temperature = self.input_temp
        
        return temperature

    cpdef exit(self):
        r"""This method deletes any file created by the software.

        Notes
        -----
        This command will also cause the Python interpreter to close
        and exit. It does not delete any forecast generated and saved
        by the software.

        See Also
        --------
        geopotential_height, relative_humidity, temperature
        """
        # Initial atmospheric conditions file.
        try:
            os.remove("initial_conditions.nc")
        except OSError:
            pass
        
        # Close Python.
        sys.exit()

# ------------------------------------------------------------------------------#

    cpdef np.ndarray pressure_thickness(self, p1=1000, p2=500):
        r"""Generates an array of the thickness of a layer.

        .. math:: h = z_2 - z_1

        Parameters
        ----------
        p1: `int`
            Bottom of the layer.
        p2: `int`
            Top of the layer.

        Returns
        -------
        `astropy.units.quantity.Quantity`
            Pressure thickness        
        
        Notes
        -----
        Pressure thickness is defined as the distance between two
        pressure surfaces. Pressure thickness is determined through the
        hypsometric equation.
        """
        # Ensure p1 is greater than p2.
        if p1 < p2:
            raise Exception("Please note that p1 must be greater than p2.")

        # Find the nearest constant pressure surface to p1 and p2 in pressure.
        cdef int indx_p1 = (np.abs(self.pressure_surfaces().value - p1)).argmin()
        cdef int indx_p2 = (np.abs(self.pressure_surfaces().value - p2)).argmin()

        pressure_thickness = (
            self.geopotential_height()[indx_p2] - self.geopotential_height()[indx_p1]
        )

        return pressure_thickness

    cpdef np.ndarray troposphere_boundaryline(self):
        r"""Generates an array of the troposphere - stratosphere boundary line.

        Returns
        -------
        `astropy.units.quantity.Quantity`
            Troposphere - stratosphere boundary line

        Notes
        -----
        The atmosphere is divided into four distinct layers: the troposphere,
        the stratosphere, the mesosphere, and the thermosphere. These layers
        are defined and  characterised by their vertical temperature profile.
        In the troposphere, temperature decreases with altitude; while in the
        stratosphere, temperature increases with altitude. This method
        determines the point at which temperature starts to increase with
        altitude (the boundary line between these two layers).

        See Also
        --------
        temperature
        """
        cdef np.ndarray temperature = self.temperature().value
        grad_temperature = np.gradient(temperature)
        cdef np.ndarray gradient_temperature = grad_temperature[0]
        gradient_temperature = np.transpose(gradient_temperature, (2, 1, 0))

        cdef np.ndarray psurface = self.pressure_surfaces().value

        cdef list trop_strat_line = []
        cdef list lat_trop_strat_line
        cdef np.ndarray temp, t
        cdef int n
        cdef float t_n, p
        for temp in gradient_temperature:
            lat_trop_strat_line = []
            for t in temp:
                n = 0
                while n < len(t):
                    t_n = t[n]
                    if t[n] >= 0:
                        p = psurface[n]
                        if p < 400:
                            lat_trop_strat_line.append(p)
                            n = len(t)
                    n += 1
            trop_strat_line.append(lat_trop_strat_line)

        troposphere_boundaryline = np.asarray(trop_strat_line)
        troposphere_boundaryline = np.transpose(troposphere_boundaryline, (1, 0)) 
        troposphere_boundaryline *= units.hPa
        return troposphere_boundaryline

# ------------------------------------------------------------------------------#

    def longitude_contourf(self, which="air_temperature", psurface=1000):
        r"""Plots a desired atmospheric parameter on a contour plot.

        Parameters
        ----------
        which: `str`
            Desired atmospheric parameter
        psurface: `int`
            Pressure at which the contour plot is generated

        Notes
        -----
        For the raw data, please see the other methods found in this class.
        This plot is layed on top of a EckertIII global projection. The
        axes are latitude and longitude. If you would like a particular
        atmospheric parameter to be added to this method, either create 
        an issue on GitHub or send an email to support@amsimp.com.

        See Also
        --------
        psurface_contourf, thickness_contourf
        """
        if self.planet == "Earth":        
            # Ensure, which, is a string.
            if not isinstance(which, str):
                raise Exception(
                    "which must be a string of the name of the"
                    + "atmospheric parameter of interest."
                )

            # Index of the nearest pressure surface in 
            # amsimp.Backend.pressure_surfaces()
            indx_psurface = (
                np.abs(self.pressure_surfaces().value - psurface)
            ).argmin()
            
            # Defines the axes, and the data.
            latitude, longitude = np.meshgrid(
                self.latitude_lines().value,
                self.longitude_lines().value
            )

            if which == "temperature" or which == "air_temperature":
                data = self.temperature()[indx_psurface, :, :]
                data_type = "Air Temperature"
                unit = " (K)"
            elif which == "geopotential_height" or which == "height":
                data = self.geopotential_height()[indx_psurface, :, :]
                data_type = "Geopotential Height"
                unit = " (m)"
            elif which == "density" or which == "atmospheric_density":
                data = self.density()[indx_psurface, :, :]
                data_type = "Atmospheric Density"
                unit = " ($\\frac{kg}{m^3}$)"
            elif which == "humidity" or which == "relative_humidity":
                data = self.relative_humidity()[indx_psurface, :, :]
                data_type = "Relative Humidity"
                unit = " (%)"
            elif which == "virtual_temperature":
                data = self.virtual_temperature()[indx_psurface, :, :]
                data_type = "Virtual Temperature"
                unit = " (K)"
            elif which == "vapor_pressure":
                data = self.vapor_pressure()[indx_psurface, :, :]
                data_type = "Vapor Pressure"
                unit = " (hPa)"
            elif which == "potential_temperature":
                data = self.potential_temperature()[indx_psurface, :, :]
                data_type = "Potential Temperature"
                unit = " (K)"
            elif which == "precipitable_water" or which == "precipitable_water_vapor":
                data = self.precipitable_water()
                data_type = "Precipitable Water Vapor"
                unit = " (mm)"
            elif which == "zonal_wind":
                data = self.wind()[0][indx_psurface, :, :]
                data_type = "Zonal Wind"
                unit = " ($\\frac{m}{s}$)"
            elif which == "meridional_wind":
                data = self.wind()[1][indx_psurface, :, :]
                data_type = "Meridional Wind"
                unit = " ($\\frac{m}{s}$)"
            elif which == "wind":
                data = self.wind()
                data = np.sqrt(data[0]**2 + data[1]**2)[indx_psurface, :, :]
                data_type = "Wind"
                unit = " ($\\frac{m}{s}$)"
            elif which == "mixing_ratio":
                data = self.mixing_ratio()[indx_psurface, :, :]
                data_type = "Mixing Ratio"
                unit = " (Dimensionless)"
            else:
                raise Exception(
                    "Invalid keyword. which must be a string of an atmospheric "
                    + "parameter included with AMSIMP."
                )

            psurfaces = self.pressure_surfaces().value
            if psurface < psurfaces[-1] or psurface > psurfaces[0]:
                raise Exception(
                    "psurface must be a real number within the isobaric boundaries."
                    + " The value of psurface was: {}".format(
                        psurface
                    )
                )

            # EckertIII projection details.
            ax = plt.axes(projection=ccrs.EckertIII())
            ax.set_global()
            ax.coastlines()
            ax.gridlines()

            # Contourf plotting.
            minimum = data.min()
            maximum = data.max()
            levels = np.linspace(minimum, maximum, 21)
            data, lon = add_cyclic_point(data, coord=self.longitude_lines().value)
            data, lat = add_cyclic_point(
                np.transpose(data), coord=self.latitude_lines().value
            )
            data = np.transpose(data)
            contour = plt.contourf(
                lon,
                lat,
                data,
                transform=ccrs.PlateCarree(),
                levels=levels,
            )

            # Add SALT.
            plt.xlabel("Latitude ($\phi$)")
            plt.ylabel("Longitude ($\lambda$)")
            if not self.input_data:
                title = (
                    data_type + " ("
                    + str(self.date.year) + '-' + str(self.date.month) + '-'
                    + str(self.date.day) + " " + str(self.date.hour)
                    + ":00 h)"
                )
            else:
                title = data_type
        
            if which != "precipitable_water" and which != "precipitable_water_vapor":
                title = (
                    title + " (" + "Pressure Surface = " + str(
                        self.pressure_surfaces()[indx_psurface]
                    ) + ")"
                )

            plt.title(title)

            # Colorbar creation.
            colorbar = plt.colorbar()
            tick_locator = ticker.MaxNLocator(nbins=15)
            colorbar.locator = tick_locator
            colorbar.update_ticks()
            colorbar.set_label(
                data_type + unit
            )

            plt.show()
            plt.close()
        else:
            raise NotImplementedError(
                "Visualisations for planetary bodies other than Earth" 
                + " is not currently implemented."
            )
    
    def psurface_contourf(self, which="air_temperature", central_long=352.3079):
        r"""Plots a desired atmospheric parameter on a contour plot, with the axes
        being latitude, and pressure surfaces. 
        
        Parameters
        ----------
        which: `str`
            Desired atmospheric parameter
        central_long: `int`
            Line of longitude at which the contour plot is generated

        Notes
        -----
        For the raw data, please see the other methods found in this class.
        This plot is layed on top of a EckertIII global projection. The
        axes are pressure and latitude. If you would like a particular
        atmospheric parameter to be added to this method, either create 
        an issue on GitHub or send an email to support@amsimp.com.

        See Also
        --------
        longitude_contourf, thickness_contourf
        """
        # Ensure, which, is a string.
        if not isinstance(which, str):
            raise Exception(
                "which must be a string of the name of the atmospheric parameter"
                + " of interest."
            )
        
        # Ensure central_long is between 0 and 359.
        if central_long < 0 or central_long > 359:
            raise Exception(
                "central_long must be a real number between 0 and 359. The value"
                + " of central_long was: {}".format(
                    central_long
                )
            )
        
        # Index of the nearest central_long in amsimp.Backend.longtitude_lines()
        indx_long = (np.abs(self.longitude_lines().value - central_long)).argmin()

        # Defines the axes, and the data.
        latitude, pressure_surfaces = np.meshgrid(
            self.latitude_lines().value, self.pressure_surfaces()
        )
        
        if which == "temperature" or which == "air_temperature":
            data = self.temperature()[:, :, indx_long]
            data_type = "Air Temperature"
            unit = " (K)"
        elif which == "density" or which == "atmospheric_density":
            data = self.density()[:, :, indx_long]
            data_type = "Atmospheric Density"
            unit = " ($\\frac{kg}{m^3}$)"
        elif which == "humidity" or which == "relative_humidity":
            data = self.relative_humidity()[:, :, indx_long]
            data_type = "Relative Humidity"
            unit = " (%)"
        elif which == "virtual_temperature":
            data = self.virtual_temperature()[:, :, indx_long]
            data_type = "Virtual Temperature"
            unit = " (K)"
        elif which == "vapor_pressure":
            data = self.vapor_pressure()[:, :, indx_long]
            data_type = "Vapor Pressure"
            unit = " (hPa)"
        elif which == "potential_temperature":
            data = self.potential_temperature()[:, :, indx_long]
            data_type = "Potential Temperature"
            unit = " (K)"
        elif which == "zonal_wind":
            data = self.wind()[0][:, :, indx_long]
            data_type = "Zonal Wind"
            unit = " ($\\frac{m}{s}$)"
        elif which == "meridional_wind":
            data = self.wind()[1][:, :, indx_long]
            data_type = "Meridional Wind"
            unit = " ($\\frac{m}{s}$)"
        elif which == "mixing_ratio":
            data = self.mixing_ratio()[:, :, indx_long]
            data_type = "Mixing Ratio"
            unit = " (Dimensionless)"
        else:
            raise Exception(
                "Invalid keyword. which must be a string of an atmospheric parameter"
                + " included with AMSIMP."
            )

        # Contourf plotting.
        minimum = data.min()
        maximum = data.max()
        levels = np.linspace(minimum, maximum, 21)
        plt.contourf(
            latitude,
            pressure_surfaces,
            data,
            levels=levels,
        )

        # Add SALT.
        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Pressure (hPa)")
        plt.yscale('log')
        plt.gca().invert_yaxis()
        if not self.input_data:
            plt.title(
                data_type + " ("
                + str(self.date.year) + '-' + str(self.date.month) + '-'
                + str(self.date.day) + " " + str(self.date.hour)
                + ":00 h, Longitude = "
                + str(np.round(self.longitude_lines()[indx_long], 2)) + ")"
            )
        else:
            plt.title(
                data_type + " ("
                + "Longitude = "
                + str(np.round(self.longitude_lines()[indx_long], 2)) + ")"
            )

        # Colorbar creation.
        colorbar = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=15)
        colorbar.locator = tick_locator
        colorbar.update_ticks()
        colorbar.set_label(
            data_type + unit
        )

        # Average boundary line between the troposphere and the stratosphere.
        troposphere_boundaryline = self.troposphere_boundaryline()
        avg_tropstratline = np.mean(troposphere_boundaryline) + np.zeros(
            len(troposphere_boundaryline)
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
        plt.close()
    
    def thickness_contourf(self, p1=1000, p2=500):
        r"""Plots the thickness of a layer on a contour plot.

        Parameters
        ----------
        p1: `int`
            Bottom of the layer
        p2: `int`
            Top of the layer

        Notes
        -----
        This plot is layed on top of a EckertIII global projection. If you
        would like a particular atmospheric parameter to be added to 
        this method, either create an issue on GitHub or send an email
        to support@amsimp.com.

        See Also
        --------
        pressure_thickness, longitude_contourf, psurface_contourf
        """
        if self.planet == "Earth":
            # Defines the axes, and the data.
            latitude, longitude = (
                self.latitude_lines().value, self.longitude_lines().value
            )

            pressure_thickness = self.pressure_thickness(p1=p1, p2=p2)

            # EckertIII projection details.
            ax = plt.axes(projection=ccrs.EckertIII())
            ax.set_global()
            ax.coastlines()
            ax.gridlines()

            # Contourf plotting.
            minimum = pressure_thickness.min()
            maximum = pressure_thickness.max()
            levels = np.linspace(minimum, maximum, 21)
            pressure_thickness, longitude = add_cyclic_point(
                pressure_thickness, coord=longitude
            )
            pressure_thickness, latitude = add_cyclic_point(
                np.transpose(pressure_thickness), coord=latitude
            )
            pressure_thickness = np.transpose(pressure_thickness)
            contour = plt.contourf(
                longitude,
                latitude,
                pressure_thickness,
                transform=ccrs.PlateCarree(),
                levels=levels,
            )

            # Index of the rain / snow line
            indx_snowline = (np.abs(levels.value - 5400)).argmin()
            contour.collections[indx_snowline].set_color('black')
            contour.collections[indx_snowline].set_linewidth(1) 

            # Add SALT.
            plt.xlabel("Latitude ($\phi$)")
            plt.ylabel("Longitude ($\lambda$)")
            if not self.input_data:
                plt.title("Pressure Thickness ("
                    + str(self.date.year) + '-' + str(self.date.month) + '-'
                    + str(self.date.day) + " " + str(self.date.hour) + ":00 h"
                    + ")"
                )
            else:
                plt.title("Pressure Thickness")

            # Colorbar creation.
            colorbar = plt.colorbar()
            tick_locator = ticker.MaxNLocator(nbins=15)
            colorbar.locator = tick_locator
            colorbar.update_ticks()
            colorbar.set_label(
                "Pressure Thickness (" + str(p1) + " hPa - " + str(p2) + " hPa) (m)"
            )

            # Footnote
            if p1 == 1000 and p2 == 500:
                plt.figtext(
                    0.99,
                    0.01,
                    "Rain / Snow Line is marked by the black line (5,400 m).",
                    horizontalalignment="right",
                )

            plt.show()
            plt.close()
        else:
            raise NotImplementedError(
                "Visualisations for planetary bodies other than Earth" 
                + " is not currently implemented."
            )
