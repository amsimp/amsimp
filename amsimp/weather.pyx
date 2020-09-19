#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
# cython: embedsignature=True, binding=True
"""
AMSIMP Weather Forecasting Class. For information about this class is
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
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
from datetime import timedelta, datetime
import os, sys
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style, ticker, gridspec
import matplotlib.animation as animation
import numpy as np
from astropy import units
from astropy import constants as constant
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cpython cimport bool
from amsimp.wind cimport Wind
from amsimp.wind import Wind
from astropy.units.quantity import Quantity
import iris
from iris.coords import DimCoord, AuxCoord
from iris.cube import Cube, CubeList
from iris import save
from progress.bar import IncrementalBar
from tqdm import tqdm
import pickle
import warnings
from zipfile import ZipFile
import requests

# -----------------------------------------------------------------------------------------#

cdef class Weather(Wind):
    """
    AMSIMP Weather Forecasting Class - Also, known as Motus Aeris @ AMSIMP.
    This class is concerned with determining the evolution of the state of
    the atmosphere through the utilisation of a long short-term memory cell.
    Weather forecasting has traditionally been done by physical models of
    the atmosphere, which are unstable to perturbations, and thus are
    inaccurate for large periods of time. Since machine learning techniques
    are more robust to perturbations, it would be logical to combine a neural
    network with a physical model. Weather forecasting is a sequential data
    problem, therefore, a recurrent neural network is the most suitable 
    option for this task. A seed is set in order to ensure reproducibility.

    Below is a list of the methods included within this class, with a short
    description of their intended purpose. Please see the relevant class methods
    for more information:

    load_historical_data ~ generates a NumPy array of the previous state
    of the atmosphere. The number of days of data which is generated can
    be specified through the utilisation of the parameter, data_size, when
    the class is initialised.
    model_prediction ~ generates a NumPy array of evolution of the
    future state of the atmosphere. This is the prediction generated
    by the long short-term memory cell by training on the generated and
    preprocessed training data.
    """

    # Ensure reproducibility.
    tf.random.set_seed(13)

    def load_historical_data(self):
        """
        Generates a NumPy array of the previous state of the atmosphere.
        The number of days of data which is generated can be specified
        through the utilisation of the parameter, data_size, when the
        class is initialised.
        """
        # Import data into the software.
        data = self.historical_data

        # Data Cubes.
        # Temperature.
        cdef np.ndarray temperature = np.asarray(
            data.extract('air_temperature')[0].data
        )
        shape_space = (
            temperature.shape[1] * temperature.shape[2] * temperature.shape[3]
        )
        temperature = temperature.reshape(
            temperature.shape[0], shape_space
        )

        # Geopotential.
        cdef np.ndarray geopotential = np.asarray(
            data.extract('geopotential')[0].data
        )
        geopotential = geopotential.reshape(
            geopotential.shape[0], shape_space
        )

        # Relative Humidity.
        cdef np.ndarray relative_humidity = np.asarray(
            data.extract('relative_humidity')[0].data
        )
        relative_humidity = relative_humidity.reshape(
            relative_humidity.shape[0], shape_space
        )

        # Wind.
        # Zonal Wind.
        cdef np.ndarray zonal_wind = np.asarray(
            data.extract('x_wind')[0].data
        )
        zonal_wind = zonal_wind.reshape(
            zonal_wind.shape[0], shape_space
        )

        # Meridional Wind.
        cdef np.ndarray meridional_wind = np.asarray(
            data.extract('y_wind')[0].data
        )
        meridional_wind = meridional_wind.reshape(
            meridional_wind.shape[0], shape_space
        )

        # Output.
        output = (
            temperature, 
            relative_humidity, 
            geopotential,
            zonal_wind,
            meridional_wind
        )
        return output

    def generate_forecast(self, save_file=False):
        """
        Explain here.
        """
        # Suppress Tensorflow warnings.
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        warnings.filterwarnings(action='ignore', category=UserWarning)

        # Define variables.
        forecast_length = self.forecast_length.value

        # Define directory.
        import amsimp.wind 
        directory = os.path.dirname(amsimp.wind.__file__)

        # Check if the PCA and SC variables are downloaded from Google Drive.
        # If not, download them.
        variables_bool = os.path.isdir(directory + "/rnn/variables")
        
        if not variables_bool:
            # Download zip file from Google Drive.
            def download_file_from_google_drive(id, destination):
                URL = "https://docs.google.com/uc?export=download"

                session = requests.Session()

                response = session.get(URL, params={'id':id}, stream=True)
                token = get_confirm_token(response)

                if token:
                    params = {'id' : id, 'confirm':token}
                    response = session.get(URL, params=params, stream=True)

                save_response_content(response, destination)    

            def get_confirm_token(response):
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        return value

                return None

            def save_response_content(response, destination):
                CHUNK_SIZE = 32768

                total_size = 3100000000
                t = tqdm(
                    total=total_size, 
                    unit='iB', 
                    unit_scale=True,
                    desc='Downloading model'
                )
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                            t.update(len(chunk))

                t.close()
            
            # Download file.
            file_id = '1T5OvdJTCLqPerOnpBW5OFi3ZwNkBZjNW'
            destination = 'variables.zip'
            download_file_from_google_drive(file_id, destination)

            # Unzip file.
            var_directory = directory + '/rnn/variables' 
            os.mkdir(var_directory)
            with ZipFile(destination, 'r') as zip:
                zip.extractall(var_directory)
            
            # Remove zip file.
            os.remove("variables.zip")

        # Load variables.
        # Temperature.
        temp_pca, temp_sc = pickle.load(
            open(directory + "/rnn/variables/temperature.pickle", "rb")
        )

        # Geopotential.
        geo_pca, geo_sc = pickle.load(
            open(directory + "/rnn/variables/geopotential.pickle", "rb")
        )

        # Relative humidity.
        rh_pca, rh_sc = pickle.load(
            open(directory + "/rnn/variables/relative_humidity.pickle", "rb")
        )

        # Zonal Wind.
        u_pca, u_sc = pickle.load(
            open(directory + "/rnn/variables/zonal_wind.pickle", "rb")
        )

        # Meridional Wind.
        v_pca, v_sc = pickle.load(
            open(directory + "/rnn/variables/meridional_wind.pickle", "rb")
        )

        # Load models.
        # Temperature.
        temp_model = tf.keras.models.load_model(
            directory + "/rnn/models/temperature"
        )

        # Geopotential.
        geo_model = tf.keras.models.load_model(
            directory + "/rnn/models/geopotential"
        )

        # Relative humidity.
        rh_model = tf.keras.models.load_model(
            directory + "/rnn/models/relative_humidity"
        )

        # Zonal Wind.
        u_model = tf.keras.models.load_model(
            directory + "/rnn/models/zonal_wind"
        )

        # Meridional Wind.
        v_model = tf.keras.models.load_model(
            directory + "/rnn/models/meridional_wind"
        )

        # Input data.
        input_data = self.load_historical_data()

        # To batch shape.
        def to_batch(input_var):
            input_var = input_var.astype(np.float32)
            input_var = input_var.reshape(
                1, input_var.shape[0], input_var.shape[1]
            )

            return input_var

        # Temperature.
        input_temperature = input_data[0]
        input_temperature = temp_sc.transform(input_temperature)
        input_temperature = temp_pca.transform(input_temperature)
        input_temperature = to_batch(input_temperature)

        # Geopotential.
        input_geopotential = input_data[2]
        input_geopotential = geo_sc.transform(input_geopotential)
        input_geopotential = geo_pca.transform(input_geopotential)
        input_geopotential = to_batch(input_geopotential)

        # Relative Humidity.
        input_humidity = input_data[1]
        input_humidity = rh_sc.transform(input_humidity)
        input_humidity = rh_pca.transform(input_humidity)
        input_humidity = to_batch(input_humidity)

        # Zonal Wind.
        input_zonalwind = input_data[3]
        input_zonalwind = u_sc.transform(input_zonalwind)
        input_zonalwind = u_pca.transform(input_zonalwind)
        input_zonalwind = to_batch(input_zonalwind)

        # Meridional Wind.
        input_meridionalwind = input_data[4]
        input_meridionalwind = v_sc.transform(input_meridionalwind)
        input_meridionalwind = v_pca.transform(input_meridionalwind)
        input_meridionalwind = to_batch(input_meridionalwind)

        # Define shape.
        it = int(forecast_length / 6)
        shape = (int(forecast_length / 2), 17, 60, 120)

        # Define outputs.
        # Temperature.
        temperature_predictions = np.zeros(shape)
        # Geopotential.
        geopotential_predictions = np.zeros(shape)
        # Relative Humidity.
        humidity_predictions = np.zeros(shape)
        # Zonal Wind.
        zonalwind_predictions = np.zeros(shape)
        # Meridional Wind.
        meridionalwind_predictions = np.zeros(shape)
        
        # Iteriate thorough.
        n = 0
        for i in tqdm(range(it), desc='Generating forecast'):
            # Temperature.
            temperature_prediction = temp_model.predict(input_temperature)
            temperature_prediction = temperature_prediction[0]
            input_temperature = input_temperature[0]
            input_temperature = np.concatenate(
                (input_temperature, temperature_prediction),
                axis=0
            )
            input_temperature = input_temperature[3:]
            input_temperature = to_batch(input_temperature)
            temperature_prediction = temp_pca.inverse_transform(
                temperature_prediction
            )
            temperature_prediction = temp_sc.inverse_transform(
                temperature_prediction
            )
            temperature_prediction = temperature_prediction.reshape(
                3, 17, 60, 120
            )
            temperature_predictions[n:n+3] = temperature_prediction

            # Geopotential.
            geopotential_prediction = geo_model.predict(input_geopotential)
            geopotential_prediction = geopotential_prediction[0]
            input_geopotential = input_geopotential[0]
            input_geopotential = np.concatenate(
                (input_geopotential, geopotential_prediction),
                axis=0
            )
            input_geopotential = input_geopotential[3:]
            input_geopotential = to_batch(input_geopotential)
            geopotential_prediction = geo_pca.inverse_transform(
                geopotential_prediction
            )
            geopotential_prediction = geo_sc.inverse_transform(
                geopotential_prediction
            )
            geopotential_prediction = geopotential_prediction.reshape(
                3, 17, 60, 120
            )
            geopotential_predictions[n:n+3] = geopotential_prediction

            # Relative Humidity.
            humidity_prediction = rh_model.predict(input_humidity)
            humidity_prediction = humidity_prediction[0]
            input_humidity = input_humidity[0]
            input_humidity = np.concatenate(
                (input_humidity, humidity_prediction),
                axis=0
            )
            input_humidity = input_humidity[3:]
            input_humidity = to_batch(input_humidity)
            humidity_prediction = rh_pca.inverse_transform(
                humidity_prediction
            )
            humidity_prediction = rh_sc.inverse_transform(
                humidity_prediction
            )
            humidity_prediction = humidity_prediction.reshape(
                3, 17, 60, 120
            )
            humidity_predictions[n:n+3] = humidity_prediction

            # Zonal Wind.
            zonalwind_prediction = u_model.predict(input_zonalwind)
            zonalwind_prediction = zonalwind_prediction[0]
            input_zonalwind = input_zonalwind[0]
            input_zonalwind = np.concatenate(
                (input_zonalwind, zonalwind_prediction),
                axis=0
            )
            input_zonalwind = input_zonalwind[3:]
            input_zonalwind = to_batch(input_zonalwind)
            zonalwind_prediction = u_pca.inverse_transform(
                zonalwind_prediction
            )
            zonalwind_prediction = u_sc.inverse_transform(
                zonalwind_prediction
            )
            zonalwind_prediction = zonalwind_prediction.reshape(
                3, 17, 60, 120
            )
            zonalwind_predictions[n:n+3] = zonalwind_prediction

            # Meridional Wind.
            meridionalwind_prediction = v_model.predict(input_meridionalwind)
            meridionalwind_prediction = meridionalwind_prediction[0]
            input_meridionalwind = input_meridionalwind[0]
            input_meridionalwind = np.concatenate(
                (input_meridionalwind, meridionalwind_prediction),
                axis=0
            )
            input_meridionalwind = input_meridionalwind[3:]
            input_meridionalwind = to_batch(input_meridionalwind)
            meridionalwind_prediction = v_pca.inverse_transform(
                meridionalwind_prediction
            )
            meridionalwind_prediction = v_sc.inverse_transform(
                meridionalwind_prediction
            )
            meridionalwind_prediction = meridionalwind_prediction.reshape(
                3, 17, 60, 120
            )
            meridionalwind_predictions[n:n+3] = meridionalwind_prediction

            # Iteriate thorough NumPy array.
            n = n + 3

        # Parameters.
        # Pressure.
        pressure = np.array(
            [
                1, 2, 5,
                10, 20, 50,
                100, 200, 300,
                400, 500, 600,
                700, 800, 900,
                950, 1000
            ]
        )
        pressure = pressure[::-1]

        # Latitude.
        latitude = [
            i
            for i in np.arange(
                -89, 89, 3
            )
        ]
        latitude = np.asarray(latitude) * -1
        latitude = latitude[::-1]

        # Longitude.
        longitude = [
            i
            for i in np.arange(
                -180, 180, 3
            )
        ]
        longitude = np.asarray(longitude)
        lons = np.split(longitude, 2)
        west_lon, east_lon = lons[0] + 360, lons[1]
        longitude = np.concatenate((east_lon, west_lon))

        # Time.
        time = np.linspace(2, forecast_length, int(forecast_length / 2))

        # Define the coordinates for the cubes. 
        # Latitude.
        lat = DimCoord(
            latitude,
            standard_name='latitude',
            units='degrees'
        )
        # Longitude
        lon = DimCoord(
            longitude,
            standard_name='longitude', 
            units='degrees'
        )
        # Pressure Surfaces.
        p = DimCoord(
            pressure,
            long_name='pressure', 
            units='hPa'
        )
        # Time.
        forecast_period = DimCoord(
            time,
            standard_name='forecast_period', 
            units='hours'
        )
        # Forecast reference time.
        ref_time = AuxCoord(
            self.date.strftime("%Y-%m-%d %H:%M:%S"),
            standard_name='forecast_reference_time'
        )

        # Define cubes.
        # Grid.
        grid_points = [
            ('forecast_period', time),
            ('pressure',  pressure),
            ('latitude',  self.latitude_lines().value),
            ('longitude', self.longitude_lines().value),                
        ]

        # Geopotential Height Cube.
        geopotential_predictions = np.split(
            geopotential_predictions, 2, axis=3
        )
        geopotential_predictions = np.concatenate(
            (geopotential_predictions[1], geopotential_predictions[0]), axis=3
        )
        height_cube = Cube(geopotential_predictions / self.g.value,
            standard_name='geopotential_height',
            units='m',
            dim_coords_and_dims=[
                (forecast_period, 0), (p, 1), (lat, 2), (lon, 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        height_cube.add_aux_coord(ref_time)
        height_cube = height_cube.interpolate(grid_points, iris.analysis.Linear())
        # Wind Cubes.
        # Zonal Wind Cube.
        zonalwind_predictions = np.split(
            zonalwind_predictions, 2, axis=3
        )
        zonalwind_predictions = np.concatenate(
            (zonalwind_predictions[1], zonalwind_predictions[0]), axis=3
        )
        u_cube = Cube(zonalwind_predictions,
            standard_name='x_wind',
            units='m s-1',
            dim_coords_and_dims=[
                (forecast_period, 0), (p, 1), (lat, 2), (lon, 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        u_cube.add_aux_coord(ref_time)
        u_cube = u_cube.interpolate(grid_points, iris.analysis.Linear())
        # Meridional Wind Cube.
        meridionalwind_predictions = np.split(
            meridionalwind_predictions, 2, axis=3
        )
        meridionalwind_predictions = np.concatenate(
            (meridionalwind_predictions[1], meridionalwind_predictions[0]), axis=3
        )
        v_cube = Cube(meridionalwind_predictions,
            standard_name='y_wind',
            units='m s-1',
            dim_coords_and_dims=[
                (forecast_period, 0), (p, 1), (lat, 2), (lon, 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        v_cube.add_aux_coord(ref_time)
        v_cube = v_cube.interpolate(grid_points, iris.analysis.Linear())
        # Wind Speed Cube.
        wind_speed = (u_cube**2 + v_cube**2) ** 0.5
        wind_speed.standard_name = 'wind_speed'
        # Temperature.
        # Air Temperature.
        temperature_predictions = np.split(
            temperature_predictions, 2, axis=3
        )
        temperature_predictions = np.concatenate(
            (temperature_predictions[1], temperature_predictions[0]), axis=3
        )
        T_cube = Cube(temperature_predictions,
            standard_name='air_temperature',
            units='K',
            dim_coords_and_dims=[
                (forecast_period, 0), (p, 1), (lat, 2), (lon, 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        T_cube.add_aux_coord(ref_time)
        T_cube = T_cube.interpolate(grid_points, iris.analysis.Linear())
        # Relative Humidity.
        humidity_predictions = np.split(
            humidity_predictions, 2, axis=3
        )
        humidity_predictions = np.concatenate(
            (humidity_predictions[1], humidity_predictions[0]), axis=3
        )
        rh_cube = Cube(humidity_predictions,
            standard_name='relative_humidity',
            units='%',
            dim_coords_and_dims=[
                (forecast_period, 0), (p, 1), (lat, 2), (lon, 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        rh_cube = rh_cube.interpolate(grid_points, iris.analysis.Linear())
        rh_cube.data[rh_cube.data > 100] = 100
        rh_cube.data[rh_cube.data < 0] = 0
        rh_cube.add_aux_coord(ref_time)

        # Create Cube list of output parameters.
        output = CubeList([
            height_cube,
            u_cube,
            v_cube,
            wind_speed,
            T_cube,
            rh_cube,
        ])

        # If specified, save the forecast in the file format, .nc.
        if save_file:
            # Establish the file name.
            filename = 'motusaeris_amsimp_' + str(self.date.year)
            filename += str(self.date.month) + str(self.date.day)
            filename += str(self.date.hour) + '.nc'

            # Save.
            save(output, filename)
        
        return output

    def visualise(
            self,
            data=None,
            plot=[
                "air_temperature",
                "geopotential_height",
                "wind_speed",
                "relative_humidity"
            ],
            psurface=1000,
            animation_time=5,
            fname='animation.mp4'
        ):
        """
        Explain here.

        Need to fix.
        """
        # Declare variable types.
        # NumPy arrays
        cdef np.ndarray time, lat, lon, longitude, data1, data2, data3, data4
        cdef np.ndarray level1, level2, level3, level4
        # Floats.
        cdef float min1, min2, min3, min4, max1, max2, max3, max4

        if self.planet == "Earth":
            # Error checking.
            # Ensure a dataset is provided.
            if data == None:
                raise Exception("The dataset for simulation must be defined.")

            # Pressure Surface.
            try:
                pressure = data[0].coords("pressure")[0].points
            except:
                pressure = data[1].coords("pressure")[0].points

            if psurface < pressure.min() or psurface > pressure.max():
                raise Exception(
                    "psurface must be a real number within the isobaric boundaries. The value of psurface was: {}".format(
                        psurface
                    )
                )

            # Index of the nearest pressure surface in amsimp.Backend.pressure_surfaces()
            indx_psurface = (np.abs(pressure - psurface)).argmin()

            # Define the forecast period.
            time = data[0].coords("forecast_period")[0].points
            time_unit = str(data[0].coords("forecast_period")[0].units)

            # Grid.
            lat = data[0].coords("latitude")[0].points
            lon = data[0].coords("longitude")[0].points
            longitude = lon

            # Style of graph.
            style.use("fivethirtyeight")

            # Define layout.
            gs = gridspec.GridSpec(2, 2)
            fig = plt.figure(figsize=(18.5, 7.5))
            fig.subplots_adjust(hspace=0.340, bottom=0.105, top=0.905)
            plt.ion()

            # Graph 1.
            ax1 = plt.subplot(gs[0, 0], projection=ccrs.EckertIII())
            label1 = plot[0]
            data1 = data.extract(label1)[0].data
            if data1.ndim == 3: 
                data1 = data1;
            else: 
                data1 = data1[:, indx_psurface, :, :];
            data1 = np.asarray(data1.tolist())
            unit1 = " (" + str(data.extract(label1)[0].metadata.units) + ")"
            label1 = label1.replace("_", " ").title()
            # Graph 2.
            ax2 = plt.subplot(gs[1, 0], projection=ccrs.EckertIII())
            label2 = plot[1]
            data2 = data.extract(label2)[0].data
            if data2.ndim == 3:
                data2 = data2;
            else: 
                data2 = data2[:, indx_psurface, :, :];
            data2 = np.asarray(data2.tolist())
            unit2 = " (" + str(data.extract(label2)[0].metadata.units) + ")"
            label2 = label2.replace("_", " ").title()
            # Graph 3.
            ax3 = plt.subplot(gs[0, 1], projection=ccrs.EckertIII())
            label3 = plot[2]
            data3 = data.extract(label3)[0].data
            if data3.ndim == 3:
                data3 = data3;
            else:
                data3 = data3[:, indx_psurface, :, :];
            data3 = np.asarray(data3.tolist())
            unit3 = " (" + str(data.extract(label3)[0].metadata.units) + ")"
            label3 = label3.replace("_", " ").title()
            # Graph 4.
            ax4 = plt.subplot(gs[1, 1], projection=ccrs.EckertIII())
            label4 = plot[3]
            data4 = data.extract(label4)[0].data[:, indx_psurface, :, :]
            if data4.ndim == 3: 
                data4 = data4; 
            else: 
                data4 = data4[:, indx_psurface, :, :];
            data4 = np.asarray(data4.tolist())
            unit4 = " (" + str(data.extract(label4)[0].metadata.units) + ")"
            label4 = label4.replace("_", " ").title()

            # Get date.
            date = data[0].coords("forecast_reference_time")[0].points[0]

            # Source of forecast (title).
            source = data[0].attributes['source']

            # Define draw function.
            def draw(frame, colorbar):
                # Plot 1.
                # EckertIII projection details.
                ax1.set_global()
                ax1.coastlines()
                ax1.gridlines()

                # Add SALT to the graph.
                ax1.set_xlabel("Longitude ($\lambda$)")
                ax1.set_ylabel("Latitude ($\phi$)")
                ax1.set_title(label1)

                # Contour plot.
                cmap1 = plt.get_cmap("hot")
                min1 = np.min(data1)
                max1 = np.max(data1)
                level1 = np.linspace(min1, max1, 21)
                data1_plot, lon = add_cyclic_point(data1, coord=longitude)
                contour1 = ax1.contourf(
                    lon,
                    lat,
                    data1_plot[frame, :, :],
                    cmap=cmap1,
                    levels=level1,
                    transform=ccrs.PlateCarree()
                )

                # Checks for a colorbar.
                if colorbar:
                    cb1 = fig.colorbar(contour1, ax=ax1)
                    tick_locator = ticker.MaxNLocator(nbins=10)
                    cb1.locator = tick_locator
                    cb1.update_ticks()
                    cb1.set_label(unit1)

                # Plot 2.
                cmap2 = plt.get_cmap("jet")
                min2 = np.min(data2)
                max2 = np.max(data2)
                level2 = np.linspace(min2, max2, 21)
                data2_plot, lon = add_cyclic_point(data2, coord=longitude)
                contour2 = ax2.contourf(
                    lon,
                    lat, 
                    data2_plot[frame, :, :], 
                    cmap=cmap2, 
                    levels=level2,
                    transform=ccrs.PlateCarree()
                )

                # EckertIII projection details.
                ax2.set_global()
                ax2.coastlines()
                ax2.gridlines()

                # Checks for a colorbar.
                if colorbar:
                    cb2 = fig.colorbar(contour2, ax=ax2)
                    tick_locator = ticker.MaxNLocator(nbins=10)
                    cb2.locator = tick_locator
                    cb2.update_ticks()
                    cb2.set_label(unit2)

                # Add SALT to the graph.
                ax2.set_xlabel("Longitude ($\lambda$)")
                ax2.set_ylabel("Latitude ($\phi$)")
                ax2.set_title(label2)

                # Plot 3.
                cmap3 = plt.get_cmap("seismic")
                min3 = np.min(data3)
                max3 = np.max(data3)
                level3 = np.linspace(min3, max3, 21)
                data3_plot, lon = add_cyclic_point(data3, coord=longitude)
                contour3 = ax3.contourf(
                    lon, 
                    lat, 
                    data3_plot[frame, :, :], 
                    cmap=cmap3, 
                    levels=level3,
                    transform=ccrs.PlateCarree()
                )

                # EckertIII projection details.
                ax3.set_global()
                ax3.coastlines()
                ax3.gridlines()

                # Checks for a colorbar.
                if colorbar:
                    cb3 = fig.colorbar(contour3, ax=ax3)
                    tick_locator = ticker.MaxNLocator(nbins=10)
                    cb3.locator = tick_locator
                    cb3.update_ticks()
                    cb3.set_label(unit3)

                # Add SALT to the graph.
                ax3.set_xlabel("Longitude ($\lambda$)")
                ax3.set_ylabel("Latitude ($\phi$)")
                ax3.set_title(label3)

                # Plot 4.
                min4 = np.min(data4)
                max4 = np.max(data4)
                level4 = np.linspace(min4, max4, 21)
                data4_plot, lon = add_cyclic_point(data4, coord=longitude)
                contour4 = ax4.contourf(
                    lon, 
                    lat, 
                    data4_plot[frame, :, :], 
                    levels=level4,
                    transform=ccrs.PlateCarree()
                )

                # EckertIII projection details.
                ax4.set_global()
                ax4.coastlines()
                ax4.gridlines()

                # Checks for a colorbar.
                if colorbar:
                    cb4 = fig.colorbar(contour4, ax=ax4)
                    tick_locator = ticker.MaxNLocator(nbins=10)
                    cb4.locator = tick_locator
                    cb4.update_ticks()
                    cb4.set_label(unit4)

                # Add SALT to the graph.
                ax4.set_xlabel("Longitude ($\lambda$)")
                ax4.set_ylabel("Latitude ($\phi$)")
                ax4.set_title(label4)

                # Title
                time_title = '%.2f' % time[frame]
                title = (
                    source + " (" + date + ", +" + time_title 
                    + time_unit + ", " + str(pressure[indx_psurface]) 
                    + " hPa)"
                )
                fig.suptitle(title)

                return fig
                
            # Define init function
            def init():
                return draw(0, colorbar=True)

            # Define animate function.
            def animate(frame):
                return draw(frame, colorbar=False)
            
            # Define animation and save it.
            frames = data1.shape[0]
            interval = np.ceil((animation_time * 1000) / frames)
            ani = animation.FuncAnimation(
                fig, 
                animate, 
                tqdm(
                    range(frames),
                    desc='Generating animation'
                ), 
                interval=interval, 
                blit=False, 
                init_func=init, 
                repeat=False
            )
            ani.save(fname)
        else:
            raise NotImplementedError(
                "Visualisations for planetary bodies other than Earth"
                + " is not currently implemented."
            )

cdef class Dynamics(Weather):
    """
    AMSIMP Dynamics Class - This class generates a simulation of tropospheric
    and stratsopheric dynamics. Predictions are made by numerically solving
    the isobaric version of the Primitive Equations (they are coupled set
    of nonlinear PDEs). The initial conditions are defined in the class
    methods of Water, Moist, and Backend. For more information on the
    initial conditions, please see those classes.
    
    Below is a list of the methods included within this class, with a short
    description of their intended purpose. Please see the relevant class methods
    for more information.

    forecast_period ~ generates the period of time over which the forecast will
    be generated for. Please also note that this method also outputs the change in time
    used. This is used in the prognostic equations. 
    simulate ~ generates a simulation of tropospheric and stratsopheric dynamics.
    visualise ~ please explain here.
    """

    def __cinit__(
            self,
            int delta_latitude=5,
            int delta_longitude=5,
            forecast_length=72, 
            delta_t=2, 
            input_date=None, 
            historical_data=None,
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
        The parameter, forecast_length, defines the length of the 
        simulation (defined in hours). Defaults to a value of 72.

        The parameter, delta_t, defines the change with respect to time
        (in minutes) defined in the simulation. Defaults to a value of 2.
        
        For more information, please refer to amsimp.Backend.__cinit__
        method.
        """
        # Declare class variables.
        super().__init__(delta_latitude)
        super().__init__(delta_longitude)
        super().__init__(forecast_length)
        self.delta_t = delta_t
        super().__init__(input_date)
        super().__init__(historical_data)
        super().__init__(input_data)
        super().__init__(height)
        super().__init__(temp)
        super().__init__(rh)
        super().__init__(u)
        super().__init__(v)
        super().__init__(constants)

        warnings.warn(
            "This class will be deprecated in a future release", 
            DeprecationWarning
        )

    cpdef tuple forecast_period(self):
        """
        Generates the period of time over which the forecast will be generated
        for. 
        
        Please also note that this method also outputs the change in time
        used. Hence, the output is a tuple.
        """
        segments = int((self.forecast_length / self.delta_t).si.value) + 1

        # Define the forecast period.
        forecast_period = np.linspace(
            0, self.forecast_length.value, segments
        ) * self.forecast_length.unit

        # Convert to seconds.
        delta_t = self.delta_t.to(units.s)

        return forecast_period, delta_t

    def __perturbations_errorcheck(self, input_perturbation):
        """
        This method is solely utilised for determining whether a user
        defined perturbation in the amsimp.Dynamics.simulate method
        is a callable function.
        """
        if input_perturbation != None:                
            # Check if first index of tuple is a function.
            if not callable(input_perturbation):
                raise Exception(
                    "perturbations must be callable functions."
                )

    def __interpolation_cube(self, input_cube, grid_points):
        """
        This method is solely utilised for interpolating a
        given cube to a different set of grid points in the
        amsimp.Dynamics.simulate method.
        """
        output = input_cube.interpolate(grid_points, iris.analysis.Linear())
        
        return output

    cpdef simulate(
            self, 
            bool save_file=False,
            perturbations_temperature=None,
            perturbations_zonalwind=None,
            perturbations_meridionalwind=None,
            perturbations_mixingratio=None
        ):
        """
        Generates a simulation of tropospheric and stratsopheric dynamics.
        Predictions are made by numerically solving the Primitive
        Equations. Depending on the parameter specifed in the initialisation
        of the class, a long short-term memory cell may be incorpated in
        the output.
        
        The Primitive Equations are a set of nonlinear partial differential
        equations that are used to approximate global atmospheric flow and
        are used in most atmospheric models. They consist of three main sets
        of balance equations: a continuity equation, conservation of
        momentum, and a thermal energy equation.

        The Lax–Friedrichs method is used to numerically solve the Primitive
        Equations within the software. It is a numerical method for the
        solution of hyperbolic partial differential equations based on
        finite differences. The method can be described as the 
        FTCS (forward in time, centered in space) scheme with a numerical
        dissipation term of 1/2. One can view the Lax–Friedrichs method as an
        alternative to Godunov's scheme, where one avoids solving a Riemann
        problem at each cell interface

        The parameter, save_file, may be utilised to save the output of
        this class. The output will be saved as a NetCDF file. These
        files can be opened by using the Iris library, which can
        be downloaded via Anaconda Cloud.

        The perturbation parameters allow the end-user to modify the
        Primtive Equations used by the software to account.
        """
        # Error checking.
        np.seterr(all='raise')

        # Ensure save_file is a boolean value.
        if not isinstance(save_file, bool):
            raise Exception(
                "save_file must be a boolean value. The value of save_file was: {}".format(
                    save_file
                )
            )
        
        # Perturbations error checking.
        # Air temperature.
        self.__perturbations_errorcheck(perturbations_temperature)
        # Zonal wind.
        self.__perturbations_errorcheck(perturbations_zonalwind)
        # Meridional wind.
        self.__perturbations_errorcheck(perturbations_meridionalwind)
        # Mixing Ratio.
        self.__perturbations_errorcheck(perturbations_mixingratio)

        # Define variables that do not vary with respect to time.
        cdef np.ndarray latitude = self.latitude_lines().value
        cdef np.ndarray longitude = self.longitude_lines().value
        cdef np.ndarray pressure = self.pressure_surfaces()
        cdef np.ndarray pressure_3d = self.pressure_surfaces(dim_3d=True)
        cdef np.ndarray interpolate_pressure = np.insert(
            pressure, 0, pressure[0] + np.abs(pressure[0] - pressure[1])
        )
        cdef time = self.forecast_period()[0]
        
        # The Coriolis parameter at various latitudes of the Earth's surface,
        # under various approximations.
        # No approximation.
        cdef np.ndarray f = self.coriolis_parameter().value / units.s
        f = self.make_3dimensional_array(parameter=f, axis=1)

        # Define the change with respect to time.
        cdef delta_t = self.forecast_period()[1]
        # Forecast length.
        cdef forecast_length = self.forecast_length.to(units.s)
        cdef int t = 0

        # Define initial conditions.
        # Gravitational Acceleration.
        cdef g = self.g
        # Geopotential Height.
        cdef np.ndarray height = self.geopotential_height()
        # Wind.
        cdef tuple wind = self.wind()
        # Zonal Wind.
        cdef np.ndarray u = wind[0]
        # Meridional Wind.
        cdef np.ndarray v = wind[1]
        # Vertical Motion.
        cdef np.ndarray omega = self.vertical_motion()
        # Temperature.
        # Air Temperature.
        cdef np.ndarray T = self.temperature()
        # Virtual Temperature.
        cdef np.ndarray T_v = self.virtual_temperature()
        # Relative Humidity.
        cdef np.ndarray rh = self.relative_humidity()
        # Mixing Ratio.
        cdef np.ndarray q = self.mixing_ratio()
        # Precipitable Water.
        cdef np.ndarray pwv = self.precipitable_water()

        # Prediction from Recurrent Neural Network.
        cdef np.ndarray prediction_ai_temp, prediction_ai_height, prediction_ai_rh
        cdef int iterator_ai
        if self.ai:
            prediction_ai = self.model_prediction()
            prediction_ai_height = prediction_ai[2] * units.m
            prediction_ai_temp = prediction_ai[0] * units.K
            prediction_ai_rh = prediction_ai[1] * units.percent
            iterator_ai = 0

        # Define extra variable types.
        # Numy Arrays.
        cdef np.ndarray T_n, q_n, u_n, v_n, height_n, A, B, C, D, E
        cdef np.ndarray RHS, e, T_c, sat_vapor_pressure, geopotential_height
        cdef np.ndarray mean_u, mean_v, mean_T, mean_q, temperature
        cdef np.ndarray virtual_temperature, zonal_wind, meridional_wind
        cdef np.ndarray static_stability, relative_humidity, mixing_ratio
        cdef np.ndarray precipitable_water, Tv_layers, z_0, z_1, z_2, h
        cdef np.ndarray Tv_layer, log_pressure_layer
        # Booleans.
        cdef bool break_loop
        # Ints / Floats.
        cdef int len_p = len(pressure)
        cdef int i
        # Tuples.
        cdef tuple shape, shape_2d

        # Create a bar to determine progress.
        max_bar = len(time)
        bar = IncrementalBar('Progress', max=max_bar)
        # Start progress bar.
        bar.next()

        # Define NumPy array for ouputs.
        shape = (len(time), len(pressure), len(latitude), len(longitude))
        shape_2d = (len(time), len(latitude), len(longitude))
        # Geopotential Height.
        geopotential_height = np.zeros(shape) * height.unit
        geopotential_height[0, :, :, :] = height
        # Temperature.
        # Air Temperature.
        temperature = np.zeros(shape) * T.unit
        temperature[0, :, :, :] = T
        T_n = T.copy()
        # Virtual Temperature.
        virtual_temperature = np.zeros(shape) * T_v.unit
        virtual_temperature[0, :, :, :] = T_v 
        # Geostrophic Wind.
        # Zonal Wind.
        zonal_wind = np.zeros(shape) * u.unit
        zonal_wind[0, :, :, :] = u
        u_n = u.copy()
        # Meridional Wind.
        meridional_wind = np.zeros(shape) * v.unit
        meridional_wind[0, :, :, :] = v
        v_n = v.copy()
        # Vertical Motion.
        vertical_motion = np.zeros(shape) * omega.unit
        vertical_motion[0, :, :, :] = omega
        # Relative Humidity.
        relative_humidity = np.zeros(shape) * rh.unit
        relative_humidity[0, :, :, :] = rh
        # Mixing Ratio.
        mixing_ratio = np.zeros(shape) * q.unit
        mixing_ratio[0, :, :, :] = q
        q_n = q.copy()
        # Precipitable Water.
        precipitable_water = np.zeros(shape_2d) * pwv.unit
        precipitable_water[0, :, :] = pwv

        # Define the initial state for the perturbation functions.
        config = Wind(
            delta_latitude=self.delta_latitude,
            delta_longitude=self.delta_longitude,
            input_data=True,
            psurfaces=self.pressure_surfaces(),
            lat=self.latitude_lines(),
            lon=self.longitude_lines(),
            height=height, 
            temp=T, 
            rh=rh,
            u=u,
            v=v,
            constants=self.constants 
        )

        # Define the coordinates for the cubes. 
        # Latitude.
        lat = DimCoord(
            latitude,
            standard_name='latitude',
            units='degrees'
        )
        # Longitude
        lon = DimCoord(
            longitude,
            standard_name='longitude', 
            units='degrees'
        )
        # Pressure Surfaces.
        p = DimCoord(
            pressure,
            long_name='pressure', 
            units='hPa'
        )
        # Time.
        forecast_period = DimCoord(
            time,
            standard_name='forecast_period', 
            units='hours'
        )
        # Forecast reference time.
        ref_time = AuxCoord(
            self.date.strftime("%Y-%m-%d %H:%M:%S"),
            standard_name='forecast_reference_time'
        )

        # Determine geopotential height of lowest constant pressure surface for
        # the integration of the Hydrostatic Equation. 
        # Define grid to interpolate onto.
        grid_points = [
            ('pressure',  interpolate_pressure.value),
            ('latitude',  latitude),
            ('longitude', longitude),
        ]

        # Define cube.
        z = Cube(height.value,
            standard_name='geopotential_height',
            units='m',
            dim_coords_and_dims=[
                (p, 0), (lat, 1), (lon, 2)
            ],
        )

        # Interpolation of geopotential height based on new grid.
        z = z.interpolate(grid_points, iris.analysis.Linear())
        z_0 = np.asarray(z.data.tolist())[0] * height.unit

        cdef int nt = 1
        try:
            while t < forecast_length.value:
                # Wind
                # Zonal Wind.
                # Determine each term in the zonal momentum equation.
                A = self.gradient_longitude(parameter=-u*u)
                B = self.gradient_latitude(parameter=-v*u)
                C = self.gradient_pressure(parameter=-omega*u)
                D = self.gradient_longitude(parameter=-g*height)
                E = f * v

                # Add any perturbations of zonal wind defined by the user.
                if perturbations_zonalwind != None:
                    perturbations = perturbations_zonalwind(config)
                else:
                    perturbations = np.zeros(u.value.shape) * E.unit

                # Sum the RHS terms and multiple by the time step.
                RHS = (A + B + C + D + E + perturbations) * delta_t
                mean_u = (
                    (
                        u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]
                    ) + (
                        u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1]
                    ) + (
                        u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2]
                    )
                ) / 6
                u_n[1:-1, 1:-1, 1:-1] = mean_u + RHS[1:-1, 1:-1, 1:-1]

                # Meridional Wind.
                # Determine each term in the meridional momentum equation.
                A = self.gradient_longitude(parameter=-u*v)
                B = self.gradient_latitude(parameter=-v*v)
                C = self.gradient_pressure(parameter=-omega*v)
                D = self.gradient_latitude(parameter=-g*height)
                E = -f * u

                # Add any perturbations of meridional wind defined by the user.
                if perturbations_meridionalwind != None:
                    perturbations = perturbations_meridionalwind(config)
                else:
                    perturbations = np.zeros(v.value.shape) * E.unit

                # Sum the RHS terms and multiple by the time step.
                RHS = (A + B + C + D + E + perturbations) * delta_t
                mean_v = (
                    (
                        v[2:, 1:-1, 1:-1] + v[:-2, 1:-1, 1:-1]
                    ) + (
                        v[1:-1, 2:, 1:-1] + v[1:-1, :-2, 1:-1]
                    ) + (
                        v[1:-1, 1:-1, 2:] + v[1:-1, 1:-1, :-2]
                    )
                ) / 6
                v_n[1:-1, 1:-1, 1:-1] = mean_v + RHS[1:-1, 1:-1, 1:-1]

                # Temperature.
                # Air Temperature.
                # Thermodynamic Equation.
                A = self.gradient_longitude(parameter=-u*T)
                B = self.gradient_latitude(parameter=-v*T)
                C = self.gradient_pressure(parameter=-omega*T)
                D = omega * ((self.R * T) / (pressure_3d * self.c_p))

                # Add any perturbations of air temperature defined by the user.
                if perturbations_temperature != None:
                    perturbations = perturbations_temperature(config)
                else:
                    perturbations = np.zeros(T.value.shape) * D.unit
                
                # Sum the RHS terms and multiple by the time step.
                RHS = (A + B + C + D + perturbations) * delta_t
                mean_T = (
                    (
                        T[2:, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]
                    ) + (
                        T[1:-1, 2:, 1:-1] + T[1:-1, :-2, 1:-1]
                    ) + (
                        T[1:-1, 1:-1, 2:] + T[1:-1, 1:-1, :-2]
                    )
                ) / 6
                T_n[1:-1, 1:-1, 1:-1] = mean_T + RHS[1:-1, 1:-1, 1:-1]

                # Mixing ratio
                # The scalar gradients of the mixing ratio.
                dq_dx = self.gradient_longitude(parameter=q)
                dq_dy = self.gradient_latitude(parameter=q)
                dq_dp = self.gradient_pressure(parameter=q)

                # Advect mixing ratio via wind.
                A = self.gradient_longitude(parameter=-u*q)
                B = self.gradient_latitude(parameter=-v*q)
                C = self.gradient_pressure(parameter=-omega*q)

                # Add any perturbations of mixing ratio defined by the user.
                if perturbations_mixingratio != None:
                    perturbations = perturbations_mixingratio(config)
                else:
                    perturbations = np.zeros(q.value.shape) * C.unit
                
                # Sum the RHS terms and multiple by the time step.
                RHS = (A + B + C + perturbations) * delta_t
                mean_q = (
                    (
                        q[2:, 1:-1, 1:-1] + q[:-2, 1:-1, 1:-1]
                    ) + (
                        q[1:-1, 2:, 1:-1] + q[1:-1, :-2, 1:-1]
                    ) + (
                        q[1:-1, 1:-1, 2:] + q[1:-1, 1:-1, :-2]
                    )
                ) / 6
                q_n[1:-1, 1:-1, 1:-1] = mean_q + RHS[1:-1, 1:-1, 1:-1]

                # Vapor pressure.
                e = pressure_3d * q_n / (0.622 + q_n)

                # Convert temperature in Kelvin to degrees centigrade.
                T_c = T_n.value - 273.15
                # Saturated vapor pressure.
                sat_vapor_pressure = 0.61121 * np.exp(
                    (
                        18.678 - (T_c / 234.5)
                    ) * (T_c / (257.14 + T_c)
                    )
                ) 
                # Add units of measurement.
                sat_vapor_pressure *= units.kPa
                sat_vapor_pressure = sat_vapor_pressure.to(units.hPa)

                # Relative Humidity.
                rh = (e.value / sat_vapor_pressure.value) * 100
                rh[rh > 100] = 100
                rh[rh < 0] = 0
                rh *= units.percent

                # Virtual Temperature.
                T_v = T_n / (
                    1 - (
                    e / pressure_3d
                    ) * (1 - 0.622)
                )

                # Geopotential Height (Hydrostatic Balance).
                # Define variables.
                z_1 = z_0
                height_n = np.zeros(height.value.shape) * height.unit

                # Interpolation of virtual temperature.
                # Define cube.
                cube = Cube(T_v.value,
                    standard_name='virtual_temperature',
                    units='K',
                    dim_coords_and_dims=[
                        (p, 0), (lat, 1), (lon, 2)
                    ],
                )
                # Interpolate.
                cube = cube.interpolate(grid_points, iris.analysis.Linear())
                # Convert to NumPy array
                Tv_layers = np.asarray(cube.data.tolist()) * T_v.unit

                for i in range(len_p):
                    # Virtual temperature layer.
                    Tv_layer = Tv_layers[i:i+2, :, :]

                    # Log pressure of layer.
                    log_pressure_layer = np.log(interpolate_pressure[i:i+2].value)

                    # Determine thickness of layer.
                    h = (
                        -self.R / self.g * np.trapz(
                            Tv_layer, x=log_pressure_layer, axis=0
                        )
                    )

                    # Determine geopotential height of constant pressure
                    # surface.
                    z_2 = z_1 + h

                    # Redefine bottom pressure surface.
                    z_1 = z_2

                    # Add to NumPy array.
                    height_n[i, :, :] = z_2
                
                # Configure the Wind class, so, that it aligns with the
                # paramaters defined by the user.
                break_loop = False
                while not break_loop:
                    try:
                        config = Wind(
                            delta_latitude=self.delta_latitude,
                            delta_longitude=self.delta_longitude,
                            input_data=True,
                            psurfaces=self.pressure_surfaces(),
                            lat=self.latitude_lines(),
                            lon=self.longitude_lines(),
                            height=height_n, 
                            temp=T_n, 
                            rh=rh,
                            u=u_n,
                            v=v_n,
                            constants=self.constants 
                        )
                        break_loop = True
                    except:
                        pass

                # Define other prognostic variables.
                # Precipitable Water.
                pwv = config.precipitable_water()

                # Add predictions to NumPy arrays.
                # Geopotential Height.
                geopotential_height[nt, :, :, :] = height_n
                height = height_n
                # Geostrophic Wind.
                # Zonal Wind.
                zonal_wind[nt, :, :, :] = u_n
                u = u_n
                # Meridional Wind.
                meridional_wind[nt, :, :, :] = v_n
                v = v_n
                # Temperature.
                # Air Temperature.
                temperature[nt, :, :, :] = T_n
                T = T_n
                # Virtual Temperature.
                virtual_temperature[nt, :, :, :] = T_v
                # Relative Humidity.
                relative_humidity[nt, :, :, :] = rh
                # Mixing Ratio.
                mixing_ratio[nt, :, :, :] = q_n
                q = q_n
                # Precipitable Water.
                precipitable_water[nt, :, :] = pwv

                # Add time step.
                t += delta_t.value
                nt += 1
                bar.next()
        except KeyboardInterrupt:
            pass

        # Cubes.
        grid_points = [
            ('forecast_period', time.value), 
            ('pressure',  pressure.value),
            ('latitude',  latitude),
            ('longitude', longitude),
        ]
        grid_points_pwv = [
            ('forecast_period', time.value),
            ('latitude',  latitude),
            ('longitude', longitude),
        ]

        # Geopotential Height Cube.
        height_cube = Cube(geopotential_height[:, 1:-1, 1:-1, 1:-1],
            standard_name='geopotential_height',
            units='m',
            dim_coords_and_dims=[
                (forecast_period, 0), (p[1:-1], 1), (lat[1:-1], 2), (lon[1:-1], 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        height_cube = self.__interpolation_cube(
            input_cube=height_cube, grid_points=grid_points
        )
        height_cube.add_aux_coord(ref_time)
        # Wind Cubes.
        # Zonal Wind Cube.
        u_cube = Cube(zonal_wind[:, 1:-1, 1:-1, 1:-1],
            standard_name='x_wind',
            units='m s-1',
            dim_coords_and_dims=[
                (forecast_period, 0), (p[1:-1], 1), (lat[1:-1], 2), (lon[1:-1], 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        u_cube = self.__interpolation_cube(
            input_cube=u_cube, grid_points=grid_points
        )
        u_cube.add_aux_coord(ref_time)
        # Meridional Wind Cube.
        v_cube = Cube(meridional_wind[:, 1:-1, 1:-1, 1:-1],
            standard_name='y_wind',
            units='m s-1',
            dim_coords_and_dims=[
                (forecast_period, 0), (p[1:-1], 1), (lat[1:-1], 2), (lon[1:-1], 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        v_cube = self.__interpolation_cube(
            input_cube=v_cube, grid_points=grid_points
        )
        v_cube.add_aux_coord(ref_time)
        # Temperature.
        # Air Temperature.
        T_cube = Cube(temperature[:, 1:-1, 1:-1, 1:-1],
            standard_name='air_temperature',
            units='K',
            dim_coords_and_dims=[
                (forecast_period, 0), (p[1:-1], 1), (lat[1:-1], 2), (lon[1:-1], 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        T_cube = self.__interpolation_cube(
            input_cube=T_cube, grid_points=grid_points
        )
        T_cube.add_aux_coord(ref_time)
        # Virtual Temperature.
        Tv_cube = Cube(virtual_temperature[:, 1:-1, 1:-1, 1:-1],
            standard_name='virtual_temperature',
            units='K',
            dim_coords_and_dims=[
                (forecast_period, 0), (p[1:-1], 1), (lat[1:-1], 2), (lon[1:-1], 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        Tv_cube = self.__interpolation_cube(
            input_cube=Tv_cube, grid_points=grid_points
        )
        Tv_cube.add_aux_coord(ref_time)
        # Relative Humidity.
        rh_cube = Cube(relative_humidity[:, 1:-1, 1:-1, 1:-1],
            standard_name='relative_humidity',
            units='%',
            dim_coords_and_dims=[
                (forecast_period, 0), (p[1:-1], 1), (lat[1:-1], 2), (lon[1:-1], 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        rh_cube = self.__interpolation_cube(
            input_cube=rh_cube, grid_points=grid_points
        )
        relative_humidity = rh_cube.data
        relative_humidity[relative_humidity > 100] = 100
        relative_humidity[relative_humidity < 0] = 0
        rh_cube = Cube(relative_humidity,
            standard_name='relative_humidity',
            units='%',
            dim_coords_and_dims=[
                (forecast_period, 0), (p, 1), (lat, 2), (lon, 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        rh_cube.add_aux_coord(ref_time)
        # Mixing Ratio.
        q_cube = Cube(mixing_ratio[:, 1:-1, 1:-1, 1:-1],
            long_name='mixing_ratio',
            dim_coords_and_dims=[
                (forecast_period, 0), (p[1:-1], 1), (lat[1:-1], 2), (lon[1:-1], 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        q_cube = self.__interpolation_cube(
            input_cube=q_cube, grid_points=grid_points
        )
        q_cube.add_aux_coord(ref_time)
        # Precipitable Water.
        pwv_cube = Cube(precipitable_water[:, 1:-1, 1:-1],
            long_name='precipitable_water',
            units='mm',
            dim_coords_and_dims=[
                (forecast_period, 0), (lat[1:-1], 1), (lon[1:-1], 2)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        pwv_cube = self.__interpolation_cube(
            input_cube=pwv_cube, grid_points=grid_points_pwv
        )
        pwv_cube.add_aux_coord(ref_time)

        # Create Cube list of output parameters.
        output = CubeList([
            height_cube,
            u_cube,
            v_cube,
            T_cube,
            Tv_cube,
            rh_cube,
            q_cube,
            pwv_cube
        ])

        # If specified, save the forecast in the file format, .nc.
        if save_file:
            # Establish the file name.
            filename = 'motusaeris_amsimp_' + str(self.date.year)
            filename += str(self.date.month) + str(self.date.day)
            filename += str(self.date.hour) + '.nc'

            # Save.
            save(output, filename)

        # Finish progress bar.
        bar.finish()

        return output
