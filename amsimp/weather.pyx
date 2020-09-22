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
        # Ensure grid is set up correctly.
        try:
            test = self.pressure_surfaces()
            test = self.latitude_lines()
            test = self.longitude_lines()
        except:
            raise

        # Data Cubes.
        # Temperature.
        input_temp = self.input_temp
        cdef np.ndarray temperature = np.asarray(input_temp.data)
        shape_space = (
            temperature.shape[1] * temperature.shape[2] * temperature.shape[3]
        )
        temperature = temperature.reshape(
            temperature.shape[0], shape_space
        )

        # Geopotential.
        input_geo = self.input_geo
        cdef np.ndarray geopotential = np.asarray(input_geo.data)
        geopotential = geopotential.reshape(
            geopotential.shape[0], shape_space
        )

        # Relative Humidity.
        input_rh = self.input_rh
        cdef np.ndarray relative_humidity = np.asarray(input_rh.data)
        relative_humidity = relative_humidity.reshape(
            relative_humidity.shape[0], shape_space
        )

        # Wind.
        # Zonal Wind.
        input_u = self.input_u
        cdef np.ndarray zonal_wind = np.asarray(input_u.data)
        zonal_wind = zonal_wind.reshape(
            zonal_wind.shape[0], shape_space
        )

        # Meridional Wind.
        input_v = self.input_v
        cdef np.ndarray meridional_wind = np.asarray(input_v.data)
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
                    desc='Downloading models'
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
        
        # Define progress bar.
        t = tqdm(total=8, desc='Preprocessing')

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
        t.update(1)

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
        t.update(1)

        # Input data.
        input_data = self.load_historical_data()
        t.update(1)

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
        t.update(1)

        # Geopotential.
        input_geopotential = input_data[2]
        input_geopotential = geo_sc.transform(input_geopotential)
        input_geopotential = geo_pca.transform(input_geopotential)
        input_geopotential = to_batch(input_geopotential)
        t.update(1)

        # Relative Humidity.
        input_humidity = input_data[1]
        input_humidity = rh_sc.transform(input_humidity)
        input_humidity = rh_pca.transform(input_humidity)
        input_humidity = to_batch(input_humidity)
        t.update(1)

        # Zonal Wind.
        input_zonalwind = input_data[3]
        input_zonalwind = u_sc.transform(input_zonalwind)
        input_zonalwind = u_pca.transform(input_zonalwind)
        input_zonalwind = to_batch(input_zonalwind)
        t.update(1)

        # Meridional Wind.
        input_meridionalwind = input_data[4]
        input_meridionalwind = v_sc.transform(input_meridionalwind)
        input_meridionalwind = v_pca.transform(input_meridionalwind)
        input_meridionalwind = to_batch(input_meridionalwind)
        t.update(1)
        t.close()

        # Define shape.
        it = int(forecast_length / 6)
        shape = (int(forecast_length / 2) + 1, 17, 60, 120)

        # Define outputs.
        # Temperature.
        temperature_predictions = np.zeros(shape)
        temperature_predictions[0] = input_data[0][-1].reshape(
            17, 60, 120
        )
        # Geopotential.
        geopotential_predictions = np.zeros(shape)
        geopotential_predictions[0] = input_data[2][-1].reshape(
            17, 60, 120
        )
        # Relative Humidity.
        humidity_predictions = np.zeros(shape)
        humidity_predictions[0] = input_data[1][-1].reshape(
            17, 60, 120
        )
        # Zonal Wind.
        zonalwind_predictions = np.zeros(shape)
        zonalwind_predictions[0] = input_data[3][-1].reshape(
            17, 60, 120
        )
        # Meridional Wind.
        meridionalwind_predictions = np.zeros(shape)
        meridionalwind_predictions[0] = input_data[4][-1].reshape(
            17, 60, 120
        )
        
        # Iteriate thorough.
        n = 1
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

        # Define progress bar.
        t = tqdm(total=7, desc='Post-processing')
        
        # Grid.
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
        latitude = np.asarray(latitude)

        # Longitude.
        longitude = [
            i
            for i in np.arange(
                -180, 180, 3
            )
        ]
        longitude = np.asarray(longitude)

        # Time.
        time = np.linspace(0, forecast_length, int(forecast_length / 2) + 1)

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
            long_name='pressure_level', 
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
            ('pressure',  self.pressure_surfaces().value),
            ('latitude',  self.latitude_lines().value),
            ('longitude', self.longitude_lines().value),                
        ]

        # Geopotential Cube.
        geo_cube = Cube(geopotential_predictions,
            standard_name='geopotential',
            units='m2 s-2',
            dim_coords_and_dims=[
                (forecast_period, 0), (p, 1), (lat, 2), (lon, 3)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        geo_cube.add_aux_coord(ref_time)
        t.update(1)
        # Geopotential Height Cube.
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
        t.update(1)
        # Wind Cubes.
        # Zonal Wind Cube.
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
        t.update(1)
        # Meridional Wind Cube.
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
        t.update(1)
        # Wind Speed Cube.
        wind_speed = (u_cube**2 + v_cube**2) ** 0.5
        wind_speed.standard_name = 'wind_speed'
        t.update(1)
        # Temperature.
        # Air Temperature.
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
        t.update(1)
        # Relative Humidity.
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
        rh_cube.data[rh_cube.data > 100] = 100
        rh_cube.data[rh_cube.data < 0] = 0
        rh_cube.add_aux_coord(ref_time)
        t.update(1)
        t.close()

        # Create Cube list of output parameters.
        output = CubeList([
            geo_cube,
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
            filename = 'motusaeris_' + str(self.date.year)
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
        """
        # Declare variable types.
        # NumPy arrays
        cdef np.ndarray time, lat, lon, longitude, data1, data2, data3, data4
        cdef np.ndarray level1, level2, level3, level4
        # Floats.
        cdef float min1, min2, min3, min4, max1, max2, max3, max4

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
                initial=1,
                desc='Generating animation'
            ), 
            interval=interval, 
            blit=False, 
            init_func=init, 
            repeat=False
        )
        ani.save(fname)
