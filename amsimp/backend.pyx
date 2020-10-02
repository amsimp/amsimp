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
from datetime import datetime, timedelta
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

cdef class Backend:
    """
    This is the base class for AMSIMP, as such, all other classes within AMSIMP
    inherit this class, either directly or indirectly.
    """

    # Meteorologically significant constants
    # Sidereal day in seconds.
    sidereal_day = (23 + (56 / 60)) * 3600 * units.s
    # Angular rotation rate of the planet.
    Upomega = ((2 * np.pi) / ((23 + (56 / 60)) * 3600))
    # Mean radius of the planet.
    a = constant.R_earth.value * constant.R_earth.unit
    # Mass of the planet.
    M = constant.M_earth.value * constant.M_earth.unit
    # The specific heat capacity on a constant pressure surface.
    c_p = 718 * (units.J / (units.kg * units.K))
    # Gravitational acceleration.
    g = 9.80665 * (units.m / (units.s ** 2))
    # Gas constant.
    R = 287.05799596 * (units.J / (units.kg * units.K))
    # Universal gravitational constant.
    G = constant.G
    G = G.value * G.unit

    def __cinit__(self, forecast_length=120, historical_data=None):
        """
        The parameter, forecast_length, defines the length of the 
        weather forecast (defined in hours). Defaults to a value of 120.
        It is currently not recommended to generate a climate forecast
        using this software, as it has not been tested for this purpose.
        This may change at some point in the future.

        The parameter, historical_data, defines the state of the atmosphere
        in the past thirty days in two-hour intervals up to the present 
        moment. The following  parameters must be defined: air
        temperature (air_temperature), zonal wind (x_wind), meridional
        wind (y_wind), geopotential, and relative 
        humidity (relative_humidity). The expected input parameter is
        an iris cube list. Each cube must have the same grid points as
        all of the other cubes. The grid must be 3 dimensional, and consist
        of: pressure, latitude, and longitude. The pressure surfaces must
        start at the lowest pressure value, and increase from that point.
        The latitude points range from -90 to 90, and the longitude points
        range from -180 to 180. The pressure grid must be named 'air_pressure',
        and the unit of pressure must be Pascal. An example dataset may be
        download using the following Google Drive link:

        https://drive.google.com/file/d/1lsZYfnqwm1hr6d7RiDHBPC9_rUx-q_FV/view?usp=sharing
        """
        # Make the aforementioned variables available else where in the class.
        self.historical_data = historical_data
        if type(forecast_length) != Quantity:
            forecast_length *= units.hr
        self.forecast_length = forecast_length

        # Ensure self.forecast_length is greater than, or equal to 1.
        if self.forecast_length.value <= 0:
            raise Exception(
                "forecast_length must be a positive number greater than, or equal to 1. "
                +  "The value of forecast_length was: {}".format(self.forecast_length)
            )

        # The date at which the initial conditions was gathered (i.e. how
        # recent the data is).
        time = self.historical_data[0][-1].coord('time')
        date = [cell.point for cell in time.cells()]
        date = date[0]
        self.date = date

        # Function to ensure that the input data is 4 dimensional and that
        # there is 30 days of atmospheric conditions in two-hour intervals.
        def dimension(input_variable):
            if np.ndim(input_variable.data) != 4:
                raise Exception(
                    "All cubes in the cube list must be a 4 dimensional."
                )
            
            if input_variable.data.shape[0] != 180:
                raise Exception(
                    "The past thirty days in two-hour intervals up to the present "
                    + "moment must be define in order to use the software."
                )

        # Check if the input is a cube list.
        if type(self.historical_data) != iris.cube.CubeList:
            raise Exception(
                "The expected input parameter is an iris cube list."
            )
        
        # Extract all of the relevant atmospheric parameters.
        try:
            # Geopotential.
            geo = self.historical_data.extract("geopotential")[0]
            dimension(geo)
            geo = geo[:, ::-1, ::-1, :]
            self.input_geo = geo
            # Air temperature.
            temp = self.historical_data.extract("air_temperature")[0]
            dimension(temp)
            temp = temp[:, ::-1, ::-1, :]
            self.input_temp = temp
            # Relative humidity.
            rh = self.historical_data.extract("relative_humidity")[0]
            dimension(rh)
            rh = rh[:, ::-1, ::-1, :]
            self.input_rh = rh
            # Zonal wind.
            u = self.historical_data.extract("x_wind")[0]
            dimension(u)
            u = u[:, ::-1, ::-1, :]
            self.input_u = u
            # Meridional wind.
            v = self.historical_data.extract("y_wind")[0]
            dimension(v)
            v = v[:, ::-1, ::-1, :]
            self.input_v = v
        except:
            raise Exception(
                "The following  parameters must be defined: air "
                + "temperature (air_temperature), zonal wind (x_wind), "
                + "meridional wind (y_wind), geopotential, and "
                + "relative humidity (relative_humidity).")
        
        # Define the grid points.
        # Pressure surfaces.
        try:
            psurfaces = geo.coords('air_pressure')[0].points
        except:
            psurfaces = geo.coords('pressure_level')[0].points
        psurfaces *= units.mbar
        self.psurfaces = psurfaces

        # Latitude.
        lat = geo.coords('latitude')[0].points
        self.lat = lat

        # Longitude.
        lon = geo.coords('longitude')[0].points
        self.lon = lon

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
        if not is_connected():
            raise Exception(
                "You must connect to the internet in order to utilise AMSIMP."
                + " Apologies for any inconvenience caused."
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
        latitude_lines = self.lat * units.deg

        # Ensure latitude points increase.
        if latitude_lines[0] > latitude_lines[-1]:
            raise Exception(
                "The latitude points appear to decrease from one point to the next."
            )

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
        longitude_lines = self.lon * units.deg

        # Ensure longitude points increase.
        if longitude_lines[0] > longitude_lines[-1]:
            raise Exception(
                "The longitude points appear to decrease from one point to the next."
            )

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
        pressure_surfaces = self.psurfaces

        # Ensure pressure surfaces decrease.
        if pressure_surfaces[0] < pressure_surfaces[-1]:
            raise Exception(
                "The pressure surfaces appear to increase from one point to the next."
            )
        
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

    cpdef geopotential_height(self, bool cube=False):
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
        geopotential_height = self.input_geo / self.g
        geopotential_height = np.asarray(
            geopotential_height[-1].data
        ) * units.m
        
        return geopotential_height

    cpdef relative_humidity(self, bool cube=False):
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
        relative_humidity = self.input_rh[-1].data
        relative_humidity = np.asarray(relative_humidity) * units.percent
        
        return relative_humidity

    cpdef temperature(self, bool cube=False):
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
        temperature = self.input_temp[-1].data
        temperature = np.asarray(temperature) * units.K
        
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
        title = (
            data_type + " ("
            + str(self.date.year) + '-' + str(self.date.month) + '-'
            + str(self.date.day) + " " + str(self.date.hour)
            + ":00 h)"
        )
    
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
        plt.title(
            data_type + " ("
            + str(self.date.year) + '-' + str(self.date.month) + '-'
            + str(self.date.day) + " " + str(self.date.hour)
            + ":00 h, Longitude = "
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
        plt.title("Pressure Thickness ("
            + str(self.date.year) + '-' + str(self.date.month) + '-'
            + str(self.date.day) + " " + str(self.date.hour) + ":00 h"
            + ")"
        )

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
