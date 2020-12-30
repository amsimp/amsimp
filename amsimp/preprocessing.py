"""
AMSIMP Preprocessing Class. For information about this classes is
described below.

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
import os
import socket
import requests
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from astropy import units
from astropy.units.quantity import Quantity
from tqdm import tqdm
import iris
import numpy as np

# ------------------------------------------------------------------------------#

class PreProcessing:
    """
    This is the preprocessing class for AMSIMP.
    """

    def __init__(
        self, forecast_length=120, amsimp_ic=True, initialisation_conditions=None
    ):
        """
        The parameter, forecast_length, defines the length of the
        weather forecast (defined in hours). Defaults to a value of 120.
        It is currently not recommended to generate a climate forecast
        using this software, as it has not been tested for this purpose.
        This may change at some point in the future.

        The parameter, amsimp_ic, is a boolean which states whether the software will
        utilise the initialisation conditions provided by AMSIMP. The initialisation
        conditions provided are from the Global Forecasting System. They are
        currently stored on AMSIMP GitHub repository. It is updated on the 1st, 7th,
        13th, and 18th hour. Currently, this option does not provide support for
        an ensemble prediction system. This, however, will be added in a future version
        of the software.

        The parameter, initialisation_conditions, defines the state of the atmosphere
        in the past thirty days in two-hour intervals up to the present
        moment. The following  parameters must be defined: 2-metre
        temperature (2m_temperature), 850 hPa temperature (air_temperature),
        850 hPa geopotential (geopotential), and total precipitation
        (total_precipitation). The expected input parameter is
        a file name. Each file must have the same grid points as
        all of the other cubes. The grid must be 2 dimensional, and have
        ha spatial resolution of 0.25 degree, which is approximately 25
        kilometres. Introplation will be invoked if this is not the case,
        which may have a negative impact on the performance of the software and
        by extension the forecast produced. The latitude points must range from
        -90 to 90, and the longitude points must range from 0 to 360.
        """
        # Suppress Tensorflow warnings.
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Make the aforementioned variables available else where in the class.
        self.amsimp_ic = amsimp_ic
        if type(forecast_length) != Quantity:
            forecast_length *= units.hr
        self.forecast_length = forecast_length.to(units.hr)
        self.initialisation_conditions = initialisation_conditions

        # Ensure self.amsimp_ic is a boolean value.
        if not isinstance(self.amsimp_ic, bool):
            raise ValueError("The parameter, amsimp_ic, must be a boolean value.")

        # Ensure self.forecast_length is greater than, or equal to 1.
        if self.forecast_length.value <= 0:
            raise ValueError(
                "The parameter, forecast_length, must be a positive number greater than, or equal to 1. "
                + "The value of forecast_length was: {}".format(self.forecast_length)
            )
        
        # Ensure self.forecast_length is a factor of 4.
        if self.forecast_length.value % 4 != 0:
            raise ValueError("The parameter, forecast_length, must be evenly divisible by four.")

        # Ensure the parameter, self.initialisation_conditions, is not defined when the
        # parameter self.amsimp_ic is defined to be true.
        if self.amsimp_ic:
            if self.initialisation_conditions != None:
                raise Exception(
                    "The parameter, initialisation_conditions, must not be defined when amsimp_ic is defined to be true."
                )

        # Error checking if self.initialisation_conditions is defined.
        if not self.initialisation_conditions == None:
            # Ensure self.initialisation_conditions is a string value.
            if not isinstance(self.initialisation_conditions, str):
                raise ValueError(
                    "The parameter, initialisation_conditions, must be a string value."
                )

            # Check if file provided to the software exists.
            if not os.path.exists(self.initialisation_conditions):
                raise FileNotFoundError(
                    "The file provided to the software could not be located."
                )

            # Output warning if file is not of the format, NetCDF.
            if not ".nc" in self.initialisation_conditions:
                raise Warning(
                    "Currently AMSIMP only officially supports the NetCDF file format."
                )

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
        if not is_connected() and amsimp_ic:
            raise Exception(
                "You must connect to the internet in order to utilise AMSIMP."
                + " Apologies for any inconvenience caused."
            )

# ------------------------------------------------------------------------------#

    def download_dataset(self):
        """
        Enter description.
        """
        # Define url.
        url = "https://github.com/amsimp/initial-conditions/raw/main/initialisation_conditions.nc"

        # Create download request.
        response = requests.get(url, stream=True)

        # Size of file.
        total_size_in_bytes = int(response.headers.get("content-length", 0))

        # Size of block.
        block_size = 1024  # 1 Kibibyte

        #  Define download progress bar.
        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc="Downloading initialisation conditions",
        )

        # Download file
        with open("initialisation_conditions.nc", "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        # Close progress bar.
        progress_bar.close()

        # Determine if the file was downloaded in its entirety.
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise Exception(
                "An unknown error occurred, which resulted in the download failing."
            )

    def load_dataset(self):
        """
        Enter description.
        """
        # Load dataset based on whether the user defined the initialisation
        # conditions, or not.
        if self.amsimp_ic:
            # Download file from the GitHub repository if necessary.
            if not os.path.exists("initialisation_conditions.nc"):
                self.download_dataset()

            # Load dataset.
            dataset = iris.load("initialisation_conditions.nc")
        else:
            # Load dataset provided by the user.
            dataset = iris.load(self.initialisation_conditions)

        return dataset

# ------------------------------------------------------------------------------#

    def lat(self):
        """
        Enter description.
        """
        lat = np.linspace(90, -90, 721)[4:-3:4]

        return lat

    def lon(self):
        """
        Enter description.
        """
        lon = np.linspace(0, 359.75, 1440)[::4]

        return lon

# ------------------------------------------------------------------------------#

    def parameter_extraction(self):
        """
        Enter description.
        """
        # Extract the relevant parameters from the dataset.
        parameters = ["t2m", "t", "z"]
        dataset = self.load_dataset().extract(parameters)

        # Ensure all parameters are present.
        if len(dataset) != 3:
            raise Exception(
                "All of the expected parameters were not present in the dataset."
            )

        # Ensure the pressure surface defined is correct if it is relevant for
        # a given parameter.
        # Air temperature at 850 hPa.
        try:
            # Retrieve DimCoord from cube.
            t850_p = dataset.extract("air_temperature")[0].coord("pressure")

            # Ensure units are in hectopascals.
            t850_p.convert_units("hPa")

            # Check if the pressure surface is at 850 hPa.
            if not int(t850_p.points[0]) == 850:
                raise Exception(
                    "The air temperature values provided are not on the correct pressure surface, which is 850 hPa."
                )
        except:
            pass

        # Geopotential at 500 hPa.
        try:
            # Retrieve DimCoord from cube.
            z500_p = dataset.extract("geopotential")[0].coord("pressure")

            # Ensure units are in hectopascals.
            z500_p.convert_units("hPa")

            # Check if the pressure surface is at 500 hPa.
            if not int(z500_p.points[0]) == 500:
                raise Exception(
                    "The geopotential values provided are not on the correct pressure surface, which is 500 hPa."
                )
        except:
            pass

        return dataset

    def interpolate_dataset(self):
        """
        Enter description.
        """
        # Define dataset.
        dataset = self.parameter_extraction()

        # Define expected coordinates.
        # Latitude.
        lat = self.lat()
        # Longitude
        lon = self.lon()

        # Check if the expected number of time steps are present.
        # If an insufficient number is present.
        if dataset[0].shape[0] < 6:
            raise ValueError(
                "Six timesteps are required in order to generate a forecast."
            )
        # If an excess number is present.
        elif dataset[0].shape[0] > 6:
            dataset = dataset[-6:]

        #  Check if longitude values contain a negative number.
        if np.min(dataset[0].coord("longitude").points) < 0:
            raise ValueError(
                "The longitude coordinate system provided is not supported. Longitude values must range from 0 to 360."
            )

        # Define grid points for interpolation.
        grid_points = [("latitude", lat), ("longitude", lon)]

        # Loop through dataset.
        for i in range(len(dataset)):
            # Interpolate dataset to the required coordinates.
            dataset[i] = dataset[i].interpolate(grid_points, iris.analysis.Linear())

        return dataset

    def normalise_dataset(self):
        """
        Enter description.
        """
        # Define dataset.
        dataset = self.interpolate_dataset()

        # Define directory.
        import amsimp.preprocessing
        directory = os.path.dirname(amsimp.preprocessing.__file__)

        # Load normalisation variables.
        # Mean.
        mean = np.load(directory + "/model/mean.npy")

        # Standard deviation.
        std = np.load(directory + "/model/std.npy")

        # Convert cube list to NumPy array.
        dataset_numpy = np.zeros(
            (
                len(dataset),
                dataset[0].shape[0],
                dataset[0].shape[1],
                dataset[0].shape[2],
            )
        )

        # Ensure units are correct.
        # 2 metre temperature (K).
        dataset[0].convert_units("K")

        # Air temperature at 850 hPa (K).
        dataset[1].convert_units("K")

        # Geopotential at 500 hPa (m2 s-2).
        dataset[2].convert_units("m2 s-2")

        # Loop through dataset.
        for i in tqdm(range(len(dataset)), desc='Interpolating dataset'):
            # Add to NumPy array.
            dataset_numpy[i] = dataset[i].data

        # Transpose.
        dataset_numpy = np.transpose(dataset_numpy, (1, 2, 3, 0))

        #  Normalise.
        dataset_numpy = (dataset_numpy - mean) / std

        # Reshape for model input.
        dataset = dataset_numpy.reshape(
            1,
            dataset[0].shape[0],
            dataset[0].shape[1],
            dataset[0].shape[2],
            len(dataset),
        )

        return dataset
