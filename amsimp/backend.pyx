#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
"""
AMSIMP Backend Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import os
from datetime import datetime
import socket
import wget
import numpy as np
from astropy import constants
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

# -----------------------------------------------------------------------------------------#


cdef class Backend:
    """
    AMSIMP Backend Class - This is the base class for AMSIMP, as such, all
    other classes within AMSIMP inherit this class, either directly or
    indirectly.

    For methods to be included in this class they must meet one of the following
    criteria:
    (1) they are considered essential components of the software.
    (2) the output of these methods are generated solely from the methods
    classified as (1).
    (3) these methods import data from the NRLMSISE-00 atmospheric model.
    This data is retrieved from the AMSIMP data repository, which is updated 
    every hour.
    (4) they don't classify nicely into any other class.
    (5) they offer a visualisation of any of the methods found in this class.

    Below is a list of the methods included within this class, with a short
    description of their intended purpose and a bracketed number signifying
    which of the above criteria they meet. Please see the relevant class methods
    for more information. Please note that the unit information with AMSIMP
    is provided by the unit module within astropy.

    latitude_lines ~ generates a NumPy array of latitude lines (1). The unit
    of measurement is degrees.
    longitude_lines ~ generates a NumPy array of longitude lines (1). The unit
    of measurement is degrees.
    altitude_level ~ generates a NumPy array of altitude levels (1). The unit
    of measurement is metres.

    coriolis_parameter ~ generates a NumPy arrray of the Coriolis parameter (2).
    The unit of measurement is radians per second.
    gravitational_acceleration ~ generates a NumPy arrray of the gravitational
    acceleration (2). The unit of measurement is metres per second squared.

    temperature ~ outputs a NumPy array of temperature (3). The unit of
    measurement is Kelvin.
    density ~ outputs a NumPy array of atmospheric density (3). The unit of
    measurement is kilograms per cubic metre.
    
    pressure ~ outputs a NumPy array of atmospheric pressure (4). The unit of
    measurement is hectopascals (millibars).
    pressure_thickness ~ outputs a NumPy array of atmospheric pressure
    thickness (4). The unit of measurement is metres.
    potential_temperature ~ outputs a NumPy array of potential temperature (4).
    The unit of measurement is Kelvin.
    exner_function ~ outputs a NumPy array of the Exner function (4). This
    method has no unit of measurement, i.e. it is dimensionless.
    troposphere_boundaryline ~ generates a NumPy array of the
    troposphere - stratosphere boundary line (4). The unit of measurement is
    metres.
    
    longitude_contourf ~ generates a contour plot for a desired atmospheric
    process, with the axes being latitude, and longitude (5).
    altitude_contourf ~ generates a contour plot for a desired atmospheric
    process, with the axes being latitude, and altitude (5).
    pressure_thickness ~ generates a contour plot for pressure thickness,
    with the axes being latitude and longitude. This plot is then layed
    on top of a EckertIII global projection (5).
    """

    # Define units of measurement for AMSIMP.
    units = units

    # Predefined Constants.
    # Angular rotation rate of Earth.
    sidereal_day = ((23 + (56 / 60)) * 3600) * units.s
    Upomega = ((2 * np.pi) / sidereal_day) * units.rad
    # Ideal Gas Constant
    R = 287 * (units.J / (units.kg * units.K))
    # Mean radius of the Earth.
    a = constants.R_earth
    a = a.value * units.m
    # Universal Gravitational Constant.
    G = constants.G
    G = G.value * G.unit
    # Mass of the Earth.
    M = constants.M_earth
    M = M.value  * M.unit
    # The specific heat capacity on a constant pressure surface for dry air.
    c_p = 1004 * (units.J / (units.kg * units.K))
    # Gravitational acceleration at the Earth's surface.
    g = 9.80665 * (units.m / (units.s ** 2)) 
    
    # Remove extra constant pressure surfaces from the temperature, and
    # geopotential height array.
    remove_psurfaces = [23, 26, 33]

    def __cinit__(self, int delta_latitude=10, int delta_longitude=10, bool remove_files=False, forecast_length=72, bool efs=True, int models=15, bool ai=True, data_size=90, epochs=200, input_date=None, bool input_data=False, geo=None, temp=None, rh=None):
        """
        The parameters, delta_latitude and delta_longitude, defines the
        horizontal resolution between grid points within the software. The
        software solely deals with atmospheric dynamics on a synoptic scale, 
        with the equations utilised within the software becoming increasingly
        inaccurate at a local sclae. The parameter values, therefore, must be
        between 5 and 30 degrees. Defaults to a value of 10 degrees. The parameter,
        remove_files is a boolean value, and when set to True, it will remove any
        file downloaded from the AMSIMP Initial Atmospheric Conditions Data 
        Repository.
        """
        # Make the aforementioned variables available else where in the class.
        self.delta_latitude = delta_latitude
        self.delta_longitude = delta_longitude
        self.remove_files = remove_files
        self.input_date = input_date
        self.input_data = input_data

        # Ensure self.delta_latitude is between 5 and 30 degrees.
        # AMSIMP solely deals with atmospheric dynamics on a synoptic scale, with the
        # equations utilised within the software becoming increasingly inaccurate
        # at a local sclae.
        if self.delta_latitude > 30 or self.delta_latitude < 5:
            raise Exception(
                "delta_latitude must be a positive integer between 5 and 30. The value of delta_latitude was: {}".format(
                    self.delta_latitude
                )
            )

        # Ensure self.delta_longitude is between 5 and 30 degrees.
        if self.delta_longitude > 30 or self.delta_longitude < 5:
            raise Exception(
                "delta_longitude must be a positive integer between 5 and 30. The value of delta_longitude was: {}".format(
                    self.delta_longitude
                )
            )

        # The date at which the initial conditions was gathered (i.e. how
        # recent the data is).
        if self.input_date == None:
            data_date = np.load(
                wget.download(
                    "https://github.com/amsimp/initial-conditions/raw/master/date.npy", 
                    bar=None,
                )
            )
            os.remove('date.npy')
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
                    "input_date must of the type 'datetime.datetime'. The value of input_date was: {}".format(
                        self.input_date
                    )
                )
            date = self.input_date
            self.date = date

        # Ensure input_data is a boolean value.
        if not isinstance(self.input_data, bool):
            raise Exception(
                "input_data must be a boolean value. The value of input_data was: {}".format(
                    self.input_data
                )
            )

        # Function to ensure that the input data is 3 dimensional.
        def dimension(input_variable):
            if np.ndim(input_variable) != 3:
                raise Exception(
                    "All input data variables (geo, rh, temp) must be a 3 dimensional."
                )

        if self.input_data == True:
            # Check if input data is 3 dimensional.
            dimension(geo)
            dimension(rh)
            dimension(temp)

            # Convert input data lists to NumPy arrays.
            geo = np.asarray(geo)
            rh = np.asarray(rh)
            temp = np.asarray(temp)

            # Add units to input variables.
            # geo variable.
            if type(geo) != Quantity:
                geo = geo * units.m
            # rh variable.
            if type(rh) != Quantity:
                rh = rh * units.percent
            # temp variable.
            if type(temp) != Quantity:
               temp = temp * units.K

        # Make the input data variables available else where in the class.
        self.input_geo = geo
        self.input_rh = rh
        self.input_temp = temp

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
                "epochs must be a integer value. The value of epochs was: {}".format(
                    self.ai
                )
            )

        # Ensure epochs is a natural number.
        if not self.epochs > 0:
            raise Exception(
                "epochs must be a integer value. The value of epochs was: {}".format(
                    self.ai
                )
            )

        # Ensure data_size is an integer value.
        if not isinstance(self.data_size, int):
            raise Exception(
                "data_size must be a integer value. The value of data_size was: {}".format(
                    self.ai
                )
            )

        # Ensure data_size is a natural number and is greater than 14.
        if not self.data_size > 14:
            raise Exception(
                "data_size must be a integer value. The value of data_size was: {}".format(
                    self.ai
                )
            )

    cpdef np.ndarray latitude_lines(self, bool f=False):
        """
        Generates a NumPy array of latitude lines.
        """
        # In order to deterrmine the Corilios force, under the beta
        # plane approximation.
        if f:
            delta_latitude = self.delta_latitude * 3
        else:
            delta_latitude = self.delta_latitude

        cdef float i 
        sh = [
            i
            for i in np.arange(-89, 0, delta_latitude)
            if i != 0
        ]
        start_nh = sh[-1] * -1
        nh = [
            i
            for i in np.arange(start_nh, 90, delta_latitude)
            if i != 0 and i != 90
        ]

        for deg in nh:
            sh.append(deg)

        # Convert list to NumPy array and add the unit of measurement.
        latitude_lines = np.asarray(sh) * units.deg

        return latitude_lines
    
    cpdef np.ndarray longitude_lines(self):
        """
        Generates a NumPy array of longitude lines.
        """
        cdef float i
        longitude_lines = [
            i
            for i in np.arange(0, 359, self.delta_longitude)
        ]

        # Convert list to NumPy array and add the unit of measurement.
        longitude_lines = np.asarray(longitude_lines) * units.deg

        return longitude_lines

    cpdef np.ndarray pressure_surfaces(self, dim_3d=False):
        """
        Generates a NumPy array of the constant pressure surfaces. This
        is the isobaric coordinate system.
        """
        # The url to the NumPy pressure surfaces file stored on the AMSIMP
        # Initial Conditions Data repository.
        url = "https://github.com/amsimp/initial-conditions/raw/master/pressure_surfaces.npy"

        # Download the NumPy file and store the NumPy array into a variable.
        try:
            pressure_surfaces = np.load("pressure_surfaces.npy")
        except FileNotFoundError:  
            psurfaces_file = wget.download(url)
            pressure_surfaces = np.load(psurfaces_file)
        
        if self.remove_files:
            os.remove("pressure_surfaces.npy")

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

        pressure_surfaces *= self.units.hPa
        return pressure_surfaces

    cdef np.ndarray gradient_x(self, parameter=None):
        """
        Explain here.
        """
        cdef np.ndarray latitude = np.radians(self.latitude_lines().value)
        parameter = np.transpose(parameter, (1, 0, 2))

        cdef int len_latitude = len(latitude)
        cdef list parameter_list = []
        cdef int n = 0
        while n < len_latitude:
            param = (1 / (self.a * np.cos(latitude[n]))) * parameter[n]

            # Store the unit.
            unit = param.unit
            unit = unit.si

            param = param.value.tolist()
            parameter_list.append(param)

            n += 1

        parameter = np.asarray(parameter_list)
        gradient_x = np.transpose(parameter, (1, 0, 2)) * unit

        return gradient_x

    cdef np.ndarray make_3dimensional_array(self, parameter=None, axis=1):
        """
        Explain here.
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

# -----------------------------------------------------------------------------------------

    cpdef np.ndarray coriolis_parameter(self, bool f=False):
        """
        Generates a NumPy arrray of the Coriolis parameter at various latitudes
        of the Earth's surface, under the f-plane approximation. The Coriolis
        parameter is defined as two times the angular rotation of the Earth by
        the sin of the latitude you are interested in.

        Equation:
            f_0 = 2 \* Upomega * sin(\phi)
        """
        coriolis_parameter = (
            2 * self.Upomega * np.sin(np.radians(self.latitude_lines(f=f)))
        )

        return coriolis_parameter

    cpdef np.ndarray rossby_parameter(self, bool f=False):
        """
        Generates a NumPy arrray of the Rossby parameter at various latitudes
        of the Earth's surface. The Rossby parameter is defined as two times
        the angular rotation of the Earth by the cosine of the latitude you are
        interested in, all over the mean radius of the Earth.

        Equation:
            beta = frac{2 \* Upomega * cos(\phi)}{a}
        """
        rossby_parameter = (
            2 * self.Upomega * np.cos(np.radians(self.latitude_lines(f=f)))
        ) / self.a

        return rossby_parameter

    cpdef np.ndarray beta_plane(self):
        """
        Generates a NumPy arrray of the Coriolis parameter at various latitudes
        of the Earth's surface, under the beta plane approximation. The Coriolis
        parameter is defined as the sum of the Coriolis parameter at a particular
        reference latitude (see amsimp.Backend.coriolis_parameter), and the
        product of the Rossby parameter at the reference latitude and the 
        meridional distance from the reference latitude.

        Equation:
            f = f_0 + beta \* y
        """
        # Define parameters
        cdef np.ndarray f_0 = self.coriolis_parameter(f=True).value
        cdef np.ndarray beta_0 = self.rossby_parameter(f=True).value
        cdef np.ndarray lat_0 = self.latitude_lines(f=True).value
        cdef np.ndarray lat = self.latitude_lines().value

        cdef int n = 0
        f_list = []
        while n < len(lat):
            # Define the nearest reference latitude line (index value).
            nearest_lat0_index = (np.abs(lat_0 - lat[n])).argmin()

            # Define the nearest reference latitude line.
            nearest_lat0 = lat_0[nearest_lat0_index]
            # Define the nearest reference Coriolis parameter.
            nearest_f0 = f_0[nearest_lat0_index]
            # Define the nearest reference Rossby parameter.
            nearest_beta0 = beta_0[nearest_lat0_index]
            
            # Calculation, and append to list.
            f = nearest_f0 + nearest_beta0 * (
                self.a.value * np.radians(lat[n] - nearest_lat0)
            )
            f_list.append(f)

            n += 1
        
        beta_plane = np.asarray(f_list) * (units.rad / units.s)
        return beta_plane

# -----------------------------------------------------------------------------------------#

    cpdef np.ndarray geopotential_height(self):
        """
        This method imports geopotential height data from a GRIB file, which is
        located in the AMSIMP Initial Conditions Data Repository on GitHub.
        The data stored within this repo is updated every six hours by amsimp-bot.
        Following which, it outputs a NumPy array in the shape of
        (len(pressure_surfaces), len(latitude_lines), len(longitude_lines)).

        Explain here.

        I strongly recommend storing the output into a variable in order to
        prevent the need to repeatly download the file. For more information,
        visit https://github.com/amsimp/initial-conditions.
        """
        if not self.input_data:
            folder = "https://github.com/amsimp/initial-conditions/raw/master/initial_conditions/"

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
                geo_cube = iris.load("initial_conditions.nc")
                geo = np.asarray(geo_cube[1].data)
            except OSError:  
                geo_file = wget.download(url)
                geo_cube = iris.load(geo_file)
                geo = np.asarray(geo_cube[1].data)

            # Ensure that the correct data was downloaded (geopotential height).
            if geo_cube[1].units != units.m:
                raise Exception("Unable to determine the geopotential height"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
            elif geo_cube[1].metadata[0] != 'geopotential_height':
                raise Exception("Unable to determine the geopotential height"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
        else:
            geo = self.input_geo.value
        
        if np.shape(geo) == (34, 181, 360):
            # Reshape the data in such a way that it matches the pressure surfaces defined in
            # the software.
            geo = np.flip(geo, axis=0)
            geo = np.delete(geo, self.remove_psurfaces, axis=0)

            # Reshape the data in such a way that it matches the latitude lines defined in
            # the software.
            geo = np.transpose(geo, (1, 2, 0))
            geo = geo[1:-1]
            geo = np.delete(geo, [89], axis=0)
            nh_geo, sh_geo = np.split(geo, 2)
            nh_lat, sh_lat = np.split(self.latitude_lines().value, 2)
            nh_geo = nh_geo[::self.delta_latitude]
            sh_startindex = int((nh_lat[-1] * -1) - 1)
            sh_geo = sh_geo[sh_startindex::self.delta_latitude]
            geopotential_height = np.concatenate((nh_geo, sh_geo))

            # Reshape the data in such a way that it matches the longitude lines defined in
            # the software.
            geopotential_height = np.transpose(geopotential_height, (2, 0, 1))
            geopotential_height = geopotential_height[:, :, ::self.delta_longitude]
            
            if self.remove_files and not self.input_data:
                os.remove("geopotential_height.nc")

            # Define the unit of measurement for geopotential height.
            geopotential_height *= units.m
        else:
            geopotential_height = geo * units.m
        
        return geopotential_height

    cpdef np.ndarray relative_humidity(self):
        """
        This method imports relative humidity data from a GRIB file, which is
        located in the AMSIMP Initial Conditions Data Repository on GitHub.
        The data stored within this repo is updated every six hours by amsimp-bot.
        Following which, it outputs a NumPy array in the shape of
        (len(pressure_surfaces), len(latitude_lines), len(longitude_lines)).

        Explain here.

        I strongly recommend storing the output into a variable in order to
        prevent the need to repeatly download the file. For more information,
        visit https://github.com/amsimp/initial-conditions.
        """
        if not self.input_data:
            folder = "https://github.com/amsimp/initial-conditions/raw/master/initial_conditions/"
            
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
                rh_cube = iris.load("initial_conditions.nc")
                rh = np.asarray(rh_cube[2].data)
            except OSError:  
                rh_file = wget.download(url)
                rh_cube = iris.load(rh_file)
                rh = np.asarray(rh_cube[2].data)

            # Ensure that the correct data was downloaded (relative humidity).
            if rh_cube[2].units != units.percent:
                raise Exception("Unable to determine the relative humidity"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
            elif rh_cube[2].metadata[0] != 'relative_humidity':
                raise Exception("Unable to determine the relative humidity"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
        else:
            rh = self.input_rh.value
        
        if np.shape(rh) == (31, 181, 360):
            # Reshape the data in such a way that it matches the pressure surfaces defined in
            # the software.
            rh = np.flip(rh, axis=0)

            # Reshape the data in such a way that it matches the latitude lines defined in
            # the software.
            rh = np.transpose(rh, (1, 2, 0))
            rh = rh[1:-1]
            rh = np.delete(rh, [89], axis=0)
            nh_rh, sh_rh = np.split(rh, 2)
            nh_lat, sh_lat = np.split(self.latitude_lines().value, 2)
            nh_rh = nh_rh[::self.delta_latitude]
            sh_startindex = int((nh_lat[-1] * -1) - 1)
            sh_rh = sh_rh[sh_startindex::self.delta_latitude]
            relative_humidity = np.concatenate((nh_rh, sh_rh))

            # Reshape the data in such a way that it matches the longitude lines defined in
            # the software.
            relative_humidity = np.transpose(relative_humidity, (2, 0, 1))
            relative_humidity = relative_humidity[:, :, ::self.delta_longitude]
            
            if self.remove_files and not self.input_data:
                os.remove("relative_humdity.nc")

            # Define the unit of measurement for relative humidity.
            relative_humidity *= units.percent
        else:
            relative_humidity = rh * units.percent

        return relative_humidity

    cpdef np.ndarray temperature(self):
        """
        This method imports temperature data from a GRIB file, which is
        located in the AMSIMP Initial Conditions Data Repository on GitHub.
        The data stored within this repo is updated every six hours by amsimp-bot.
        Following which, it outputs a NumPy array in the shape of
        (len(pressure_surfaces), len(latitude_lines), len(longitude_lines)).

        Temperature is defined as the mean kinetic energy density of molecular
        motion.

        I strongly recommend storing the output into a variable in order to
        prevent the need to repeatly download the file. For more information,
        visit https://github.com/amsimp/initial-conditions.
        """
        if not self.input_data:
            folder = "https://github.com/amsimp/initial-conditions/raw/master/initial_conditions/"
            
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
                temp_cube = iris.load("initial_conditions.nc")
                temp = np.asarray(temp_cube[0].data)
            except OSError:  
                temp_file = wget.download(url)
                temp_cube = iris.load(temp_file)
                temp = np.asarray(temp_cube[0].data)

            # Ensure that the correct data was downloaded (temperature).
            if temp_cube[0].units != units.K:
                raise Exception("Unable to determine the temperature"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
            elif temp_cube[0].metadata[0] != 'air_temperature':
                raise Exception("Unable to determine the temperature"
                + " at this time. Please contact the developer for futher"
                + " assistance.")
        else:
            temp = self.input_temp.value
        
        if np.shape(temp) == (34, 181, 360):
            # Reshape the data in such a way that it matches the pressure surfaces defined in
            # the software.
            temp = np.flip(temp, axis=0)
            temp = np.delete(temp, self.remove_psurfaces, axis=0)

            # Reshape the data in such a way that it matches the latitude lines defined in
            # the software.
            temp = np.transpose(temp, (1, 2, 0))
            temp = temp[1:-1]
            temp = np.delete(temp, [89], axis=0)
            nh_temp, sh_temp = np.split(temp, 2)
            nh_lat, sh_lat = np.split(self.latitude_lines().value, 2)
            nh_temp = nh_temp[::self.delta_latitude]
            sh_startindex = int((nh_lat[-1] * -1) - 1)
            sh_temp = sh_temp[sh_startindex::self.delta_latitude]
            temperature = np.concatenate((nh_temp, sh_temp))

            # Reshape the data in such a way that it matches the longitude lines defined in
            # the software.
            temperature = np.transpose(temperature, (2, 0, 1))
            temperature = temperature[:, :, ::self.delta_longitude]
            
            if self.remove_files and not self.input_data:
                os.remove("temperature.nc")

            # Define the unit of measurement for temperature.
            temperature *= units.K
        else:
            temperature = temp * units.K

        return temperature

    cpdef remove_all_files(self):
        """
        Explain here.
        """
        # Pressure surfaces file.
        try:
            os.remove("pressure_surfaces.npy")
        except OSError:
            pass

        # Initial atmospheric conditions file.
        try:
            os.remove("initial_conditions.nc")
        except OSError:
            pass

        return "Deleted!"

# -----------------------------------------------------------------------------------------#

    cpdef np.ndarray gravitational_acceleration(self):
        """
        Generates a NumPy arrray of the effective gravitational acceleration
        according to WGS84 at a distance z from the Globe. The output is
        in the shape of (len(latitude_lines), len(altitude_level())). There
        is no longitudinal variation in gravitational accleration.
        """
        cdef list lat = (np.radians(self.latitude_lines().value)).tolist()
        cdef float a = 6378137
        cdef float b = 6356752.3142
        cdef float g_e = 9.7803253359
        cdef float g_p = 9.8321849378

        cdef list lat_long = []
        cdef int len_longitude = len(self.longitude_lines())
        cdef int n = 0
        while n < len_longitude:
            lat_long.append(lat)

            n += 1
        lat_long = (np.transpose(lat_long)).tolist()

        cdef list lat_list = []
        cdef int len_psurfaces = len(self.pressure_surfaces().value)
        n = 0
        while n < len_psurfaces:
            lat_list.append(lat_long)

            n += 1
        cdef np.ndarray latitude = np.asarray(lat_list)

        # Magnitude of the effective gravitational acceleration according to WGS84
        # at point P on the ellipsoid.
        g_0 = (
            (a * g_e * (np.cos(latitude) ** 2)) + (b * g_p * (np.sin(latitude) ** 2))
        ) / np.sqrt(
            ((a ** 2) * (np.cos(latitude) ** 2) + (b ** 2) * (np.sin(latitude) ** 2))
        )

        cdef float f = (a - b) / a
        cdef float m = (
            (self.Upomega.value ** 2) * (a ** 2) * b
            ) / (self.G.value * self.M.value)

        # Magnitude of the effective gravitational acceleration according to WGS84 at
        # a distance z from the ellipsoid.
        cdef np.ndarray height = self.geopotential_height().value
        gravitational_acceleration = g_0 * (
            1
            - (2 / a) * (1 + f + m - 2 * f * (np.sin(latitude) ** 2)) * height
            + (3 / (a ** 2)) * (height ** 2)
        )

        # Add the unit of measurement
        gravitational_acceleration *= (units.m / (units.s ** 2))
        return gravitational_acceleration

    cpdef np.ndarray pressure_thickness(self, p1=1000, p2=500):
        """
        Explain here.

        Pressure thickness is defined as the distance between two
        pressure surfaces.
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
        """
        Generates a NumPy array of the troposphere - stratosphere
        boundary line in the shape (len(longitude_lines), len(latitude_lines).
        This is calculated by looking at the vertical temperature profile in
        the method, amsimp.Backend.temperature().
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

# -----------------------------------------------------------------------------------------#

    def longitude_contourf(self, which=0, psurface=1000):
        """
        Plots a desired atmospheric process on a contour plot, with the axes
        being latitude and longitude. This plot is then layed on top of a 
        EckertIII global projection. For the raw data, please see the other
        methods found in this class.

        For a temperature contour plot, the value of which is 0.
        For a geopotential height contour plot, the value of which is 1.
        For a density contour plot, the value of which is 2.
        For a relative humidity, the value of which is 3.
        For a virtual temperature contour plot, the value of which is 4.
        For a vapor pressure contour plot, the value of which is 5.
        For a potential temperature contour plot, the value of which is 6.
        """
        info = " For a temperature contour plot, the value of which is 0. "
        info += "For a geopotential height contour plot, the value of which is 1. For a "
        info += "density contour plot, the value of which is 2. For a relative humidity "
        info += "contour plot, the value of which is 3. For a virtual temperature contour "
        info += "plot, the value of which is 4. For a vapor pressure contour plot, the "
        info += "value of which is 5. For a potential temperature contour plot, the "
        info += "value of which is 6."

        # Ensure, which, is a number between 0 and 6.
        if which < 0 or which > 6:
            raise Exception(
                "which must be a natural number between 0 and 6. The value of which was: {}.".format(
                    which
                ) + info
            )
        
        # Ensure, which, is a integer.
        if not isinstance(which, int):
            raise Exception(
                "which must be a natural number between 0 and 2. The value of which was: {}.".format(
                    which
                ) + info
            )

        # Ensure psurface is between 1000 and 100 hPa above sea level.
        if psurface < 1 or psurface > 1000:
            raise Exception(
                "psurface must be a real number between 1 and 1,000. The value of psurface was: {}".format(
                    psurface
                )
            )

        # Index of the nearest pressure surface in amsimp.Backend.pressure_surfaces()
        indx_psurface = (np.abs(self.pressure_surfaces().value - psurface)).argmin()
        
        # Defines the axes, and the data.
        latitude, longitude = np.meshgrid(self.latitude_lines(),
         self.longitude_lines()
        )
        if which == 0:
            data = self.temperature()[indx_psurface, :, :]
            data_type = "Temperature"
            unit = " (K)"
        elif which == 1:
            data = self.geopotential_height()[indx_psurface, :, :]
            data_type = "Geopotential Height"
            unit = " (m)"
        elif which == 2:
            data = self.density()[indx_psurface, :, :]
            data_type = "Atmospheric Density"
            unit = " ($\\frac{kg}{m^3}$)"
        elif which == 3:
            data = self.relative_humidity()[indx_psurface, :, :]
            data_type = "Relative Humidity"
            unit = " (%)"
        elif which == 4:
            data = self.virtual_temperature()[indx_psurface, :, :]
            data_type = "Virtual Temperature"
            unit = " (K)"
        elif which == 5:
            data = self.vapor_pressure()[indx_psurface, :, :]
            data_type = "Vapor Pressure"
            unit = " (hPa)"
        elif which == 6:
            data = self.potential_temperature()[indx_psurface, :, :]
            data_type = "Potential Temperature"
            unit = " (K)"

        # EckertIII projection details.
        ax = plt.axes(projection=ccrs.EckertIII())
        ax.set_global()
        ax.coastlines()
        ax.gridlines()

        # Contourf plotting.
        minimum = data.min()
        maximum = data.max()
        levels = np.linspace(minimum, maximum, 21)
        data, lon = add_cyclic_point(data, coord=self.longitude_lines())
        contour = plt.contourf(
            lon,
            self.latitude_lines(),
            data,
            transform=ccrs.PlateCarree(),
            levels=levels,
        )

        # Add SALT.
        plt.xlabel("Latitude ($\phi$)")
        plt.ylabel("Longitude ($\lambda$)")
        if not self.input_data:
            plt.title(
                data_type + " ("
                + str(self.date.year) + '-' + str(self.date.month) + '-'
                + str(self.date.day) + " " + str(self.date.hour)
                + ":00 h, Pressure Surface = "
                + str(self.pressure_surfaces()[indx_psurface]) +")"
            )
        else:
            plt.title(
                data_type + " ("
                + "Pressure Surface = "
                + str(self.pressure_surfaces()[indx_psurface]) +")"
            )

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
    
    def psurface_contourf(self, which=0, central_long=-7.6921):
        """
        Plots a desired atmospheric process on a contour plot,
        with the axes being latitude, and pressure surfaces. For the raw
        data, please see the other methods found in this class.
        If you would like a particular atmospheric process to
        be added to this method, either create an issue on
        GitHub or send an email to support@amsimp.com.

        For a temperature contour plot, the value of which is 0.
        """
        info = " For a temperature contour plot, the value of which is 0. "

        # Ensure, which, is equal to 0.
        if which < 0 or which > 0:
            raise Exception(
                "which must be equal to zero. The value of which was: {}.".format(
                    which
                ) + info
            )
        
        # Ensure, which, is a integer.
        if not isinstance(which, int):
            raise Exception(
                "which must be equal to zero. The value of which was: {}.".format(
                    which
                ) + info
            )
        
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
        latitude, pressure_surfaces = np.meshgrid(self.latitude_lines(), self.pressure_surfaces())
        
        if which == 0:
            data = self.temperature()[:, :, indx_long]
            data_type = "Temperature"
            unit = " (K)"
            cmap = plt.get_cmap("hot")

        # Contourf plotting.
        minimum = data.min()
        maximum = data.max()
        levels = np.linspace(minimum, maximum, 21)
        plt.contourf(
            latitude,
            pressure_surfaces,
            data,
            cmap=cmap,
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
        """
        Plots pressure thickness on a contour plot, with the axes being
        latitude and longitude. This plot is then layed on top of a EckertIII
        global projection. For the raw data, please use the
        amsimp.Backend.pressure_thickness() method.
        """
        # Defines the axes, and the data.
        latitude, longitude = self.latitude_lines(), self.longitude_lines()

        cdef np.ndarray pressure_thickness = self.pressure_thickness(p1=p1, p2=p2)

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
                + str(self.date.day) + " " + str(self.date.hour) + ":00 h" + ")"
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
