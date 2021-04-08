"""
AMSIMP Preprocessing Class. For information about this class is
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
from datetime import datetime, timedelta
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


class Preprocessing:
    """
    This is the preprocessing class for AMSIMP.
    """

    def __init__(
        self,
        forecast_length=120,
        amsimp_ic=True,
        initialisation_conditions=None,
        use_efs=False,
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
        ha spatial resolution of 1 degree, which is approximately 100
        kilometres. Introplation will be invoked if this is not the case,
        which may have a negative impact on the performance of the software and
        by extension the forecast produced. The latitude points must range from
        -90 to 90, and the longitude points must range from 0 to 360.
        """
        # Suppress Tensorflow warnings.
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        # Make the aforementioned variables available else where in the class.
        self.amsimp_ic = amsimp_ic
        if type(forecast_length) != Quantity:
            forecast_length *= units.hr
        self.forecast_length = forecast_length.to(units.hr)
        self.initialisation_conditions = initialisation_conditions
        self.use_efs = use_efs

        # Ensure self.amsimp_ic is a boolean value.
        if not isinstance(self.amsimp_ic, bool):
            raise ValueError("The parameter, amsimp_ic, must be a boolean value.")

        # Ensure self.forecast_length is greater than, or equal to 1.
        if self.forecast_length.value <= 0:
            raise ValueError(
                "The parameter, forecast_length, must be a positive number greater than, or equal to 1. "
                + "The value of forecast_length was: {}".format(self.forecast_length)
            )

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

        # Ensure use_efs is a boolean value.
        if not isinstance(self.use_efs, bool):
            raise ValueError("The parameter, use_efs, must be a boolean value.")

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

    def __download_file(self, url, desc):
        r"""Generates and downloads a cube with the required parameter.

        Parameters
        ----------
        url : `str`
            The url of the required parameter to download
        desc : `str`
            The name of the required parameter to download

        Returns
        -------
        `iris.cube.Cube`
            Cube of the downloaded parameter

        Notes
        -----
        This method is activated when the parameter, amsimp_ic, is defined to
        be true. The data is downloaded from the AMSIMP Initial Conditions
        repository. This is intended as a private method, and may not function
        correctly if used.

        See Also
        --------
        load_dataset
        """
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
            desc="Downloading {} initialisation condition".format(desc),
        )

        # Download file
        with open("temp.nc", "wb") as file:
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

        # Load parameter from dataset.
        parameter = iris.load("temp.nc")[0]
        parameter.data

        # Remove temporary file.
        os.remove("temp.nc")

        return parameter

    def load_dataset(self):
        r"""Generates a cube list with the required dataset loaded, either the
        file provided is loaded or the files from the AMSIMP Initial Conditions
        repository are downloaded and saved into a single file.

        Returns
        -------
        `iris.cube.CubeList`
            Cube list with the required dataset loaded

        Notes
        -----
        The AMSIMP Initial Conditions repository is update four times daily, at
        1 am, 7 am, 1 pm, and 7 pm. The near real-time initialisation conditions
        are provided by the National Oceanic and Atmospheric Adminstrations'
        Global Data Assimilation System (GDAS).
        """
        # Determine whether the type is determinstic or an ensemble spread.
        if self.use_efs:
            type_init = "gefs"
        else:
            type_init = "gfs"

        # Function to download files and save appropriately.
        def download_files():
            # Air temperature.
            t = self.__download_file(
                "https://github.com/amsimp/initial-conditions/raw/main/{}/t.nc".format(
                    type_init
                ),
                "air temperature",
            )

            # Geopotential.
            z = self.__download_file(
                "https://github.com/amsimp/initial-conditions/raw/main/{}/z.nc".format(
                    type_init
                ),
                "geopotential",
            )

            # Define dataset.
            dataset = iris.cube.CubeList([t, z])

            # Save dataset.
            iris.save(dataset, "{}.nc".format(type_init))

        # Load dataset based on whether the user defined the initialisation
        # conditions, or not.
        if self.amsimp_ic:
            # Download file from the GitHub repository if necessary.
            if not os.path.exists("{}.nc".format(type_init)):
                # Download files.
                download_files()
            else:
                # Load dataset.
                dataset = iris.load("{}.nc".format(type_init))

                # Define dates.
                dataset_date = (
                    dataset[0]
                    .coord("time")
                    .units.num2date(dataset[0].coord("time").points[0])
                )
                old = datetime.now() - timedelta(days=1)

                # Check if the dataset is old and delete if it is.
                # Following which download the new dataset.
                if old > dataset_date:
                    # Remove old files.
                    os.remove("{}.nc".format(type_init))

                    # Download files.
                    download_files()

            # Load dataset.
            dataset = iris.load("{}.nc".format(type_init))
        else:
            # Load dataset provided by the user.
            dataset = iris.load(self.initialisation_conditions)

        return dataset

    # ------------------------------------------------------------------------------#

    def lat(self):
        r"""Generates an array of latitude lines in accordance with the shape
        expected by the operational model of the AMSIMP Global Forecast Model.

        Returns
        -------
        `numpy.ndarray`
            Latitude lines

        Notes
        -----
        The resolution of this model is approximately 100 kilometres (1 degree).

        See Also
        --------
        lon
        """
        # Define directory.
        import amsimp.preprocessing

        directory = os.path.dirname(amsimp.preprocessing.__file__)

        # Retrieve lines of latitude.
        lat = np.load(directory + "/gfm/lat.npy")

        return lat

    def lon(self):
        r"""Generates an array of longitude lines in accordance with the shape
        expected by the operational model of the AMSIMP Global Forecast Model.

        Returns
        -------
        `numpy.ndarray`
            Longitude lines

        Notes
        -----
        The resolution of this model is approximately 100 kilometres (1 degree).

        See Also
        --------
        lat
        """
        # Define directory.
        import amsimp.preprocessing

        directory = os.path.dirname(amsimp.preprocessing.__file__)

        # Retrieve lines of longitude.
        lon = np.load(directory + "/gfm/lon.npy")

        return lon

    # ------------------------------------------------------------------------------#

    def parameter_extraction(self):
        r"""Generates a cube list with the expected parameters extracted in the
        expected order for interplolation and normalisation.

        Returns
        -------
        `iris.cube.CubeList`
            Cube list with expected parameters extracted

        Notes
        -----
        The parameters, in this order, are: air temperature at 2 metres above
        the surface, air temperature at a pressure surface of 850 hectopascals,
        and geopotential at a pressure surface of 500 hectopascals.

        See Also
        --------
        load_dataset, interpolate_dataset, normalise_dataset
        """
        # Extract the relevant parameters from the dataset.
        parameters = ["t", "z"]
        dataset = self.load_dataset().extract(parameters)

        # Ensure all parameters are present.
        if len(dataset) != 2:
            raise Exception(
                "All of the expected parameters were not present in the dataset."
            )

        return dataset

    def interpolate_dataset(self):
        r"""Generates a cube list with the expected parameters, interpolated
        if necessary onto the grid required for input into the operational AMSIMP
        Global Forecast Model.

        Returns
        -------
        `iris.cube.CubeList`
            Cube list with interpolated grid for operational model

        Notes
        -----
        This method also ensures the expected number of time steps are
        included, and raises an error when an insufficient number is present.
        The number of time steps required is 6.

        See Also
        --------
        load_dataset, parameter_extraction, normalise_dataset
        """
        # Define dataset.
        dataset = self.parameter_extraction()

        # Define expected coordinates.
        # Latitude.
        lat = self.lat()
        # Longitude
        lon = self.lon()

        # Check if longitude values contain a negative number.
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
        r"""Generates a NumPy array with the expected parameters, normalised,
        processed onto the grid required for input into the operational AMSIMP
        Global Forecast Model.

        Returns
        -------
        `numpy.ndarray`
            Normalised and preprocessed dataset for forecast model input

        Notes
        -----
        This method also converts the parameters into the correct units of
        measurement if it is necessary to do so.

        See Also
        --------
        load_dataset, parameter_extraction, interpolate_dataset
        """
        # Define dataset.
        dataset = self.interpolate_dataset()

        # Define directory.
        import amsimp.preprocessing

        directory = os.path.dirname(amsimp.preprocessing.__file__)

        # Load normalisation variables.
        # Mean.
        mean = np.load(directory + "/gfm/mean.npy")

        # Standard deviation.
        std = np.load(directory + "/gfm/std.npy")

        # Convert cube list to NumPy array.
        if self.use_efs:
            # Ensemble forecast system.
            dataset_numpy = np.zeros(
                (
                    len(dataset),
                    dataset[0].shape[0],
                    dataset[0].shape[1],
                    dataset[0].shape[2],
                )
            )
        else:
            # Deterministic forecast system.
            dataset_numpy = np.zeros(
                (len(dataset), dataset[0].shape[0], dataset[0].shape[1])
            )

        # Ensure units are correct.
        # Air temperature (K).
        dataset[0].convert_units("K")

        # Geopotential (m2 s-2).
        dataset[1].convert_units("m2 s-2")

        # Loop through dataset.
        for i in tqdm(range(len(dataset)), desc="Interpolating dataset"):
            # Add to NumPy array.
            dataset_numpy[i] = dataset[i].data

        # Transpose.
        if self.use_efs:
            # Ensemble forecast system.
            dataset = np.transpose(dataset_numpy, (1, 2, 3, 0))
        else:
            # Deterministic forecast system.
            dataset = np.transpose(dataset_numpy, (1, 2, 0))

            # Reshape for model input.
            dataset = dataset.reshape(
                1,
                dataset.shape[0],
                dataset.shape[1],
                dataset.shape[2],
            )

        # Normalise.
        dataset = (dataset - mean) / std

        return dataset
