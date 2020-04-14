#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
"""
AMSIMP Dynamics Class. For information about this class is described below.
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies
from datetime import timedelta
import os
import wget
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import numpy as np
import cartopy.crs as ccrs
from cpython cimport bool
from amsimp.wind cimport Wind
from amsimp.wind import Wind
from astropy.units.quantity import Quantity
import iris
from iris.coords import DimCoord
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.cube import CubeList
from iris import save
from progress.bar import IncrementalBar
from metpy.calc import smooth_gaussian
from numpy.random import random_sample

# -----------------------------------------------------------------------------------------#

cdef class RNN(Wind):
    """
    Detailed explanation.
    """

    # Feature Scaling 
    sc = MinMaxScaler(feature_range=(0,1))

    def download_historical_data(self):
        """
        Explain here.
        """
        # Folder containing historical data on GitHub.
        folder = "https://github.com/amsimp/initial-conditions/raw/master/initial_conditions/"

        # Data lists.
        # Temperature.
        cdef list T_list = []

        # Geopotential Height.
        cdef list geo_list = []

        # Relative Humidity.
        cdef list rh_list = []

        # Define variable types.
        cdef np.ndarray T, geo, rh

        # Beginning date of the dataset.
        cdef date = self.date
        date = date + timedelta(days=-self.data_size)

        # Download progresss.
        max_bar = self.data_size * 4
        bar = IncrementalBar('Downloading Historical Data', max=max_bar)

        for i in range(max_bar):
            # Define the date in terms of integers.
            day = date.day
            month = date.month
            year = date.year
            hour = date.hour

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

            # File url.
            file_url = folder+year+"/"+month+"/"+day+"/"+hour+"/"
            file_url += "initial_conditions.nc"

            # Download file.
            download = wget.download(file_url, bar=None)

            # Load file.
            data = iris.load(download)

            # Convert data to NumPy arrays
            # Temperature.
            T = np.asarray(data[0].data)
            # Geopotential Height.
            geo = np.asarray(data[1].data)
            # Relative Humidity.
            rh = np.asarray(data[2].data)

            # Configure the Wind class, so, that it aligns with the
            # paramaters defined by the user.
            config = Wind(
                delta_latitude=self.delta_latitude,
                delta_longitude=self.delta_longitude,
                remove_files=self.remove_files,
                input_data=True, 
                geo=geo, 
                temp=T, 
                rh=rh, 
            )

            # Redefine NumPy arrays.
            # Temperature.
            T = config.temperature().value
            T = T.flatten()
            # Geopotential Height.
            geo = config.geopotential_height().value
            geo = geo.flatten()
            # Relative Humidity.
            rh = config.relative_humidity().value
            rh = rh.flatten()

            # Append to list.
            # Temperature.
            T_list.append(T)
            # Geopotential Height.
            geo_list.append(geo)
            # Relative Humidity.
            rh_list.append(rh)

            # Add six hours to date to get the next dataset.
            date = date + timedelta(hours=+6)

            # Remove download file.
            os.remove(download)

            # Increment progress bar.
            bar.next()

        # Convert lists to NumPy arrays.
        # Temperature.
        temperature = np.asarray(T_list)
        # Geopotential Height.
        geopotential_height = np.asarray(geo_list)
        # Relative Humidity.
        relative_humidity = np.asarray(rh_list)

        # Finish bar.
        bar.finish()

        # Output.
        output = (temperature, geopotential_height, relative_humidity)
        return output        

    def standardise_data(self):
        """
        Explain here.
        """
        # Define atmospheric parameters.
        historical_data = self.download_historical_data()
        temperature = historical_data[0]
        geopotential_height = historical_data[1]
        relative_humidity = historical_data[2]

        # Training / Validation split.
        split = np.floor((np.shape(temperature)[0]) * 0.9)
        split = int(split)

        # Standardise the data.
        # Temperature.
        temperature = self.sc.fit_transform(temperature)
        # Geopotential Height.
        geopotential_height = self.sc.fit_transform(geopotential_height)
        # Relative Humidity.
        relative_humidity = self.sc.fit_transform(relative_humidity)

        # Output.
        output = (temperature, geopotential_height, relative_humidity)
        return output, split

    def preprocess_data(
        self, dataset, past_history, future_target
    ):
        """
        Explain here.
        """
        X, y = list(), list()
        for i in range(len(dataset)):
            # Find the end.
            end_ix = i + past_history
            out_end_ix = end_ix + future_target
            
            # Determine if we are beyond the dataset.
            if out_end_ix > len(dataset):
                break
            
            # Gather the input and output components.
            seq_x, seq_y = dataset[i:end_ix, :], dataset[end_ix:out_end_ix, :]

            # Append to list.
            X.append(seq_x)
            y.append(seq_y)

        return np.asarray(X), np.asarray(y)

    def model_prediction(self):
        """
        Explain here.
        """
        # Dataset.
        input_from_method = self.standardise_data()
        dataset = input_from_method[0]

        # Temperature.
        temperature = dataset[0]
        # Geopotential Height.
        geopotential_height = dataset[1]
        # Relative Humidity.
        relative_humidity = dataset[2]

        # Training / Validation split.
        split = input_from_method[1]

        # Batch size.
        batch_size = 5

        # The network is shown data from the last 15 days.
        past_history = 15 * 4

        # The network predicts the next 7 days worth of steps.
        future_target = 7 * 4

        # The dataset is preprocessed.
        # Temperature.
        x_temp, y_temp = self.preprocess_data(
            temperature, past_history, future_target
        )
        # Geopotential Height.
        x_geo, y_geo = self.preprocess_data(
            geopotential_height, past_history, future_target
        )
        # Relative Humidity.
        x_rh, y_rh = self.preprocess_data(
            relative_humidity, past_history, future_target
        )

        # Number of features.
        features = x_temp.shape[2]

        # Create, and train models.
        # Temperature model.
        # Create.
        temp_model = Sequential()
        temp_model.add(LSTM(
            200, activation='relu', input_shape=(past_history, features)
        ))
        temp_model.add(RepeatVector(future_target))
        temp_model.add(LSTM(200, activation='relu', return_sequences=True))
        temp_model.add(TimeDistributed(Dense(features)))
        temp_model.compile(optimizer='adam', loss='mse')
        # Train.
        temp_model.fit(
            x_temp, y_temp, epochs=self.epochs, verbose=1, batch_size=batch_size
        )

        # Geopotential height model.
        # Create.
        geo_model = Sequential()
        geo_model.add(LSTM(
            200, activation='relu', input_shape=(past_history, features)
        ))
        geo_model.add(RepeatVector(future_target))
        geo_model.add(LSTM(200, activation='relu', return_sequences=True))
        geo_model.add(TimeDistributed(Dense(features)))
        geo_model.compile(optimizer='adam', loss='mse')
        # Train.
        geo_model.fit(
            x_geo, y_geo, epochs=self.epochs, verbose=1, batch_size=batch_size
        )

        # Relative Humidity model.
        # Create.
        rh_model = Sequential()
        rh_model.add(LSTM(
            200, activation='relu', input_shape=(past_history, features)
        ))
        rh_model.add(RepeatVector(future_target))
        rh_model.add(LSTM(200, activation='relu', return_sequences=True))
        rh_model.add(TimeDistributed(Dense(features)))
        rh_model.compile(optimizer='adam', loss='mse')
        # Train.
        rh_model.fit(
            x_rh, y_rh, epochs=self.epochs, verbose=1, batch_size=batch_size
        )

        # Prediction.
        # Set up inputs.
        predict_temp_input = temperature[-past_history]
        predict_geo_input = geopotential_height[-past_history]
        predict_rh_input = relative_humidity[-past_history]
        # Make predictions.
        predict_temp = temp_model.predict(predict_temp_input, verbose=1)
        predict_geo = geo_model.predict(predict_geo_input, verbose=1)
        predict_rh = rh_model.predict(predict_rh_input, verbose=1)

        # Invert normalisation.
        predict_temp = self.sc.inverse_transform(predict_temp)
        predict_geo = self.sc.inverse_transform(predict_geo)
        predict_rh = self.sc.inverse_transform(predict_rh)

        # Reshape into 3d arrays.
        # Dimensions.
        len_time = len(predict_temp)
        len_pressure = len(self.pressure_surfaces())
        len_lat = len(self.latitude_lines())
        len_lon = len(self.longitude_lines())
        # Reshape.
        predict_temp = predict_temp.reshape(len_time, len_pressure, len_lat, len_lon)
        predict_geo = predict_geo.reshape(len_time, len_pressure, len_lat, len_lon)
        predict_rh = predict_rh.reshape(len_time, len_pressure, len_lat, len_lon)

        return predict_temp, predict_geo, predict_rh

cdef class Dynamics(Wind):
    """
    AMSIMP Dynamics Class - Also, known as Motus Aeris @ AMSIMP. This class
    generates rudimentary simulation of tropospheric and stratsopheric
    dynamics on a synoptic scale. Predictions are made by numerically
    solving the Primitive Equations (they are PDEs). The initial conditions
    are defined in the class methods of Water, Wind, and Backend. For more
    information on the initial conditions, please see those classes. The
    boundary conditions are handled by the gradient function between
    NumPy.

    Below is a list of the methods included within this class, with a short
    description of their intended purpose. Please see the relevant class methods
    for more information.

    forecast_temperature ~ this method outputs the forecasted temperature
    for the specified number of forecast days. Every single day is divided
    into hours, meaning the length of the outputted list is 24 times the number
    of forecast days specified.
    forecast_density ~ this method outputs the forecasted atmospheric
    density for the specified number of forecast days. Every single day is
    divided into hours, meaning the length of the outputted list is 24 times the
    number of forecast days specified.
    forecast_pressure ~ this method outputs the forecasted atmospheric
    pressure for the specified number of forecast days. Every single day is
    divided into hours, meaning the length of the outputted list is 24 times the
    number of forecast days specified.
    forecast_pressurethickness ~ this method outputs the forecasted pressure
    thickness for the specified number of forecast days. Every single day is
    divided into hours, meaning the length of the outputted list is 24 times the
    number of forecast days specified.
    forecast_precipitablewater ~ this method outputs the forecasted precipitable
    water for the specified number of forecast days. Every single day is divided
    into hours, meaning the length of the outputted list is 24 times the number
    of forecast days specified.

    simulate ~ this method outputs a visualisation of how temperature, pressure
    thickness, geostrophic wind, and precipitable water vapor will evolve for
    the specified number of forecast days.
    """

    def __cinit__(self, int delta_latitude=10, int delta_longitude=10, bool remove_files=False, forecast_length=72, bool efs=True, int models=15, bool ai=True, bool input_data=False, geo=None, temp=None, rh=None):
        """
        Defines the length of the forecast (in hours) generated in the simulation.
        This value must be greater than 0, and less than 168 in order
        to ensure that the simulation methods function correctly. Defaults to
        a value of 72.

        For more information, please refer to amsimp.Backend.__cinit__()
        method.
        """
        # Add units to forecast length variable.
        if type(forecast_length) != Quantity:
            forecast_length = forecast_length * self.units.h

        # Declare class variables.
        super().__init__(delta_latitude)
        super().__init__(delta_longitude)
        super().__init__(remove_files)
        self.forecast_length = forecast_length
        self.efs = efs
        self.ai = ai
        super().__init__(input_data)
        super().__init__(geo)
        super().__init__(temp)
        super().__init__(rh)

        # Ensure self.forecast_length is between 0 and 168.
        if self.forecast_length.value > 168 or self.forecast_length.value <= 0:
            raise Exception(
                "forecast_length must be a positive integer between 1 and 168. The value of forecast_length was: {}".format(
                    self.forecast_length
                )
            )

        # Ensure efs is a boolean value.
        if not isinstance(self.efs, bool):
            raise Exception(
                "efs must be a boolean value. The value of efs was: {}".format(
                    self.efs
                )
            )

        # If efs is disabled, ensure only one model is run.
        if self.efs:
            self.models = models
        else:
            self.models = 1

        # Ensure models is an integer value.
        if not isinstance(self.models, int):
            raise Exception(
                "models must be a integer value. The value of integer was: {}".format(
                    self.ai
                )
            )

        # Ensure models is a natural number.
        if not self.models > 0:
            raise Exception(
                "models must be a integer value. The value of integer was: {}".format(
                    self.ai
                )
            )
        
        # Ensure ai is a boolean value.
        if not isinstance(self.ai, bool):
            raise Exception(
                "ai must be a boolean value. The value of ai was: {}".format(
                    self.ai
                )
            )

    def forecast_period(self):
        """
        Explain here.
        """
        forecast_length = int(self.forecast_length.value)

        # Define the forecast period.
        forecast_period = np.linspace(
            0, forecast_length, (forecast_length * 30) + 1
        ) * self.forecast_length.unit

        # Change in time (delta t) utilised to get a numerical solution to the
        # partial derivative equations.
        delta_t = (1/30) * self.units.hr
        delta_t = delta_t.to(self.units.s)

        return forecast_period, delta_t

    cpdef atmospheric_prognostic_method(self, bool save_file=False, p1=1000, p2=500):
        """
        Explain here.
        """
        # Ensure p1 is greater than p2.
        if p1 < p2:
            raise Exception("Please note that p1 must be greater than p2.")
        
        # Ensure save_file is a boolean value.
        if not isinstance(save_file, bool):
            raise Exception(
                "save_file must be a boolean value. The value of save_file was: {}".format(
                    save_file
                )
            )

        # Define variables that do not vary with respect to time.
        cdef lat = self.latitude_lines().value
        cdef np.ndarray latitude = np.radians(lat)
        cdef np.ndarray longitude = np.radians(self.longitude_lines().value)
        cdef np.ndarray pressure = self.pressure_surfaces()
        cdef np.ndarray pressure_3d = self.pressure_surfaces(dim_3d=True)
        cdef time = self.forecast_period()[0]
        cdef np.ndarray g = self.gravitational_acceleration()
        
        # The Coriolis parameter at various latitudes of the Earth's surface,
        # under the beta plane approximation.
        cdef np.ndarray f = self.beta_plane().value / self.units.s
        f = self.make_3dimensional_array(parameter=f, axis=1)

        # The Coriolis parameter at various reference latitudes
        # of the Earth's surface, under the beta plane approximation.
        cdef np.ndarray f_0 = self.coriolis_parameter(f=True).value
        cdef np.ndarray lat_0 = self.latitude_lines(f=True).value
        cdef int n = 0
        cdef f0_list = []
        while n < len(latitude):
            # Define the nearest reference latitude line (index value).
            nearest_lat0_index = (np.abs(lat_0 - lat[n])).argmin()

            # Define the nearest reference latitude line.
            nearest_lat0 = lat_0[nearest_lat0_index]
            # Define the nearest reference Coriolis parameter.
            nearest_f0 = f_0[nearest_lat0_index]

            f0_list.append(nearest_f0)

            n += 1
        f_0 = np.asarray(f0_list) / self.units.s
        f_0 = self.make_3dimensional_array(parameter=f_0, axis=1)

        # Define the change with respect to time.
        cdef delta_t = self.forecast_period()[1]
        cdef delta_2t = delta_t * 2
        cdef delta_halfstep = delta_t / 2
        # Forecast length.
        cdef forecast_length = self.forecast_period()[0][-1].to(self.units.s)
        cdef int t

        # Define initial conditions.
        # Geopotential Height.
        cdef np.ndarray height = self.geopotential_height()
        # Geopotential.
        cdef np.ndarray geo = height * g
        cdef np.ndarray geo_i = geo
        cdef np.ndarray geo_initial = geo
        # Geostrophic Wind.
        # Zonal Wind.
        cdef np.ndarray u_g = self.zonal_wind()
        # Meridional Wind.
        cdef np.ndarray v_g = self.meridional_wind()
        # Vertical Motion.
        cdef np.ndarray omega = self.vertical_motion()
        # Static Stability.
        cdef np.ndarray sigma = self.static_stability()
        # Temperature.
        # Air Temperature.
        cdef np.ndarray T = self.temperature()
        cdef np.ndarray T_i = T
        cdef np.ndarray T_initial = T
        # Virtual Temperature.
        cdef np.ndarray T_v = self.virtual_temperature()
        cdef np.ndarray Tv_i = T_v
        cdef np.ndarray Tv_initial = T_v
        # Relative Humidity.
        cdef np.ndarray rh = self.relative_humidity()
        # Thickness.
        cdef np.ndarray thickness = self.pressure_thickness(p1=p1, p2=p2)
        # Precipitable Water.
        cdef np.ndarray pwv = self.precipitable_water()

        # Index of pressure surfaces.
        cdef int indx_p1 = (np.abs(self.pressure_surfaces().value - p1)).argmin()
        cdef int indx_p2 = (np.abs(self.pressure_surfaces().value - p2)).argmin()

        # Create a bar to determine progress.
        max_bar = len(time) * self.models
        bar = IncrementalBar('Progress', max=max_bar)
        # Start progress bar.
        bar.next()

        # Ensemble Forecast System (EFS)
        # Define lists.
        # Geopotential Height.
        cdef list HeightList = []
        # Geostrophic Wind.
        # Zonal Wind.
        cdef list ZonalList = []
        # Meridional Wind.
        cdef list MeridionalList = []
        # Vertical Motion.
        cdef list VerticalList = []
        # Static Stability.
        cdef list StabilityList = []
        # Temperature.
        # Air Temperature.
        cdef list TemperatureList = []
        # Virtual Temperature.
        cdef list VirtualList = []
        # Relative Humidity.
        cdef list HumidityList = []
        # Pressure Thickness.
        cdef list ThicknessList = []
        # Precipitable Water.
        cdef list WaterList = []

        # Acceptable amount of devivation from initial conditions.
        # Geopotential Height.
        cdef np.ndarray geo_max = np.max(np.max(geo_i, axis=2), axis=1)
        cdef np.ndarray geo_mean = np.max(np.mean(geo_i, axis=2), axis=1)
        cdef np.ndarray geo_dev = self.make_3dimensional_array((geo_max - geo_mean), axis=0)
        # Temperature.
        cdef np.ndarray T_max = np.max(np.max(T_i, axis=2), axis=1)
        cdef np.ndarray T_mean = np.max(np.mean(T_i, axis=2), axis=1)
        cdef np.ndarray T_dev = self.make_3dimensional_array((T_max - T_mean), axis=0)
        # Virtual Temperature.
        cdef np.ndarray Tv_max = np.max(np.max(Tv_i, axis=2), axis=1)
        cdef np.ndarray Tv_mean = np.max(np.mean(Tv_i, axis=2), axis=1)
        cdef np.ndarray Tv_dev = self.make_3dimensional_array((Tv_max - Tv_mean), axis=0)

        # Model runs.
        cdef int model_run = self.models
        cdef int m = 0

        # Prediction from Recurrent Neural Network.
        prediction_ai = RNN.model_prediction()
        cdef np.ndarray prediction_ai_temp = prediction_ai[0] * self.units.K
        cdef np.ndarray prediction_ai_height = prediction_ai[1] * self.units.m
        cdef np.ndarray prediction_ai_rh = prediction_ai[2] * self.units.percent
        cdef int iterator_ai = 0

        # Define variable types.
        # Numy Arrays.
        cdef np.ndarray geo_0, geo_n, T_0, T_n, Tv_0, Tv_n
        cdef np.ndarray dv_dx, du_dy, geostrophic_vorticity, A, A_diffx, A_diffy, Part_1
        cdef np.ndarray dT_dx, dT_dy, nabla_T, B, dB_dp, Part_2, RHS_geo
        cdef np.ndarray change_geopotential, u_0, v_0, C, RHS,
        cdef np.ndarray dTv_dx, dTv_dy, e, T_c, sat_vapor_pressure
        cdef np.ndarray randomise, geo_rand, T_rand, Tv_rand
        # Lists.
        cdef list geopotential_height, temperature, virtual_temperature
        cdef list zonal_wind, meridional_wind, vertical_motion, static_stability 
        cdef list relative_humidity, pressure_thickness, precipitable_water

        while m < model_run:
            # Smoothing operator (filter with normal distribution
            # of weights) on initial conditions.
            geo_i = smooth_gaussian(
                scalar_grid=geo_i.value,
                n=3,
            ).magnitude * geo_i.unit
            T_i = smooth_gaussian(
                scalar_grid=T_i.value,
                n=3,
            ).magnitude * T.unit
            Tv_i = smooth_gaussian(
                scalar_grid=Tv_i.value,
                n=3,
            ).magnitude * Tv_i.unit

            # Define lists for ouputs.
            # Geopotential Height.
            geopotential_height = []
            geopotential_height.append(geo_i.value)
            # Temperature.
            # Air Temperature.
            temperature = []
            temperature.append(T_i.value)
            # Virtual Temperature.
            virtual_temperature = []
            virtual_temperature.append(Tv_i.value)
            # Geostrophic Wind.
            # Zonal Wind.
            zonal_wind = []
            zonal_wind.append(u_g.value)
            # Meridional Wind.
            meridional_wind = []
            meridional_wind.append(v_g.value)
            # Vertical Motion.
            vertical_motion = []
            vertical_motion.append(omega.value)
            # Static Stability.
            static_stability = []
            static_stability.append(sigma.value)
            # Relative Humidity.
            relative_humidity = []
            relative_humidity.append(rh.value)
            # Pressure Thickness.
            pressure_thickness = []
            pressure_thickness.append(thickness.value)
            # Precipitable Water.
            precipitable_water = []
            precipitable_water.append(pwv.value)

            # For initial state.
            n = 0
            # Forecast using the models initial conditions.
            t = 0
            while t < forecast_length.value:
                # Define initial state.
                # Geopotential.
                if n > 2:
                    geo_0 = geo
                    geo = geo_n

                # Temperature.
                if n > 2:
                    T_0 = T
                    T = T_n
                
                # Virtual temperature.
                if n > 2:
                    Tv_0 = T_v
                    T_v = Tv_n

                # The Prognostic Section for Geopotential Height (Height
                # Tendency Equation).
                # Geostrophic Vorticity.
                dv_dx = self.gradient_x(
                    parameter=np.gradient(
                        v_g, longitude, axis=2
                    )
                )
                du_dy = np.gradient(u_g, self.a * latitude, axis=1)
                geostrophic_vorticity = dv_dx - du_dy
                
                # Part (1).
                A = geostrophic_vorticity + f
                A_diffx = self.gradient_x(
                    parameter=np.gradient(
                        A, longitude, axis=2
                    )
                )
                A_diffy = np.gradient(A, self.a * latitude, axis=1)
                Part_1 = f_0 * (
                    (-u_g * A_diffx) + (-v_g * A_diffy)
                )

                # The derivative of temperature with respect to latitude
                # and longitude.
                dT_dx = self.gradient_x(
                    parameter=np.gradient(
                        T, longitude, axis=2
                    )
                )
                dT_dy = np.gradient(T, self.a * latitude, axis=1)

                # Part (2)
                nabla_T = -u_g * dT_dx + -v_g * dT_dy
                
                B = (self.R / pressure_3d) * nabla_T
                dB_dp = np.gradient(B, pressure, axis=0)

                Part_2 = ((f_0 ** 2) / sigma) * dB_dp

                RHS_geo = Part_1 - Part_2
                RHS_geo *= -1

                # Change with respect to time for geopotential.
                if n == 0:
                    RHS_geo = RHS_geo * delta_halfstep
                else:
                    RHS_geo = RHS_geo * delta_2t
                change_geopotential = RHS_geo.value * geo.unit

                # Smoothing operator (filter with normal distribution
                # of weights).
                change_geopotential = smooth_gaussian(
                    scalar_grid=change_geopotential.value,
                    n=14,
                ).magnitude * change_geopotential.unit

                if n == 0:
                    geo = geo_i + change_geopotential
                elif n == 1:
                    geo = geo_i + change_geopotential
                elif n == 2:
                    geo_n = geo_i + change_geopotential
                else:
                    geo_n = geo_0 + change_geopotential
                    # Apply Robert-Asselin time filter.
                    geo = geo + 0.1 * (geo_n - 2*geo + geo_0)
                
                # Convert to geopotential height.
                height = geo / g

                # Pressure thickness.
                thickness = height[indx_p2] - height[indx_p1]

                # Temperature.
                # Air Temperature.
                # Determine the mean flow to linearise the PDE.
                u_0 = np.mean(u_g)
                v_0 = np.mean(v_g)

                # Thermodynamic Equation.
                A = -u_0 * dT_dx
                B = -v_0 * dT_dy
                C = (pressure_3d / self.R) * sigma * omega
                
                if n == 0:
                    RHS = (A + B + C) * delta_halfstep
                else:
                    RHS = (A + B + C) * delta_2t

                # Smoothing operator (filter with normal distribution
                # of weights).
                RHS = smooth_gaussian(
                    scalar_grid=RHS.value,
                    n=14,
                ).magnitude * RHS.unit

                if n == 0:
                    T = T_i + RHS
                elif n == 1:
                    T = T_i + RHS
                elif n == 2:
                    T_n = T_i + RHS
                else:
                    T_n = T_0 + RHS
                    # Apply Robert-Asselin time filter.
                    T = T + 0.1 * (T_n - 2*T + T_0)

                # Virtual Temperature
                # The derivative of virtual temperature with
                # respect to latitude and longitude.
                dTv_dx = self.gradient_x(
                    parameter=np.gradient(
                        T_v, longitude, axis=2
                    )
                )
                dTv_dy = np.gradient(T_v, self.a * latitude, axis=1)

                # Thermodynamic Equation.
                A = -u_0 * dTv_dx
                B = -v_0 * dTv_dy
                C = (pressure_3d / self.R) * sigma * omega
                
                if n == 0:
                    RHS = (A + B + C) * delta_halfstep
                else:
                    RHS = (A + B + C) * delta_2t
                
                # Smoothing operator (filter with normal distribution
                # of weights).
                RHS = smooth_gaussian(
                    scalar_grid=RHS.value,
                    n=14,
                ).magnitude * RHS.unit

                if n == 0:
                    T_v = Tv_i + RHS
                elif n == 1:
                    T_v = Tv_i + RHS
                elif n == 2:
                    Tv_n = Tv_i + RHS
                else:
                    Tv_n = Tv_0 + RHS
                    # Apply Robert-Asselin time filter.
                    T_v = T_v + 0.1 * (Tv_n - 2*T_v + Tv_0)

                # Vapor pressure.
                e = (
                    (
                        (pressure_3d) / (1 - 0.622)
                    ) - (
                        (pressure_3d * T) / (T_v * (1 - 0.622))
                    )
                )

                # Convert temperature in Kelvin to degrees centigrade.
                T_c = T.value - 273.15
                # Saturated vapor pressure.
                sat_vapor_pressure = 0.61121 * np.exp(
                    (
                        18.678 - (T_c / 234.5)
                    ) * (T_c / (257.14 + T_c)
                    )
                ) 
                # Add units of measurement.
                sat_vapor_pressure *= self.units.kPa
                sat_vapor_pressure = sat_vapor_pressure.to(self.units.hPa)

                # Relative Humidity.
                rh = (e.value / sat_vapor_pressure.value) * 100
                rh[rh > 100] = 100
                rh[rh < 0] = 0
                rh *= self.units.percent

                # Recurrent Neural Network.
                if self.ai:
                    # Apply predictions every six hours.
                    if t != 0 and t % 21600 == 0:
                        # Temperature. 
                        T = (T + prediction_ai_temp) / 2
                        # Relative Humidity.
                        rh = (rh + prediction_ai_rh) / 2
                        # Geopotential Height.
                        height = (height + prediction_ai_height) / 2
                        # Geopotential.
                        geo = height * g
                
                # Configure the Wind class, so, that it aligns with the
                # paramaters defined by the user.
                config = Wind(
                    delta_latitude=self.delta_latitude,
                    delta_longitude=self.delta_longitude,
                    remove_files=self.remove_files,
                    input_data=True, 
                    geo=height, 
                    temp=T, 
                    rh=rh, 
                )

                # Recurrent Neural Network.
                if self.ai:
                    # Apply predictions every six hours.
                    if t != 0 and t % 21600 == 0:
                        # Virtual Temperature.
                        T_v = self.virtual_temperature()

                # Geostrophic Wind.
                # Zonal Wind.
                u_g = config.zonal_wind()
                # Meridional Wind.
                v_g = config.meridional_wind()

                # Vertical Motion.
                omega = config.vertical_motion()

                # Static Stability.
                sigma = config.static_stability()

                # Precipitable Water.
                pwv = config.precipitable_water()

                # Append predictions to list.
                if n > 1:
                    # Geopotential Height.
                    geopotential_height.append(height.value)
                    # Geostrophic Wind.
                    # Zonal Wind.
                    zonal_wind.append(u_g.value)
                    # Meridional Wind.
                    meridional_wind.append(v_g.value)
                    # Vertical Motion.
                    vertical_motion.append(omega.value)
                    # Static Stability
                    static_stability.append(sigma.value)
                    # Temperature.
                    # Air Temperature.
                    temperature.append(T.value)
                    # Virtual Temperature.
                    virtual_temperature.append(T_v.value)
                    # Relative Humidity.
                    relative_humidity.append(rh.value)
                    # Thickness.
                    pressure_thickness.append(thickness.value)
                    # Precipitable Water.
                    precipitable_water.append(pwv.value)

                    # Add time step.
                    t += delta_t.value
                    bar.next()
                
                n += 1

            # Append model lists to the relevant EFS list.
            HeightList.append(geopotential_height)
            ZonalList.append(zonal_wind)
            MeridionalList.append(meridional_wind)
            VerticalList.append(vertical_motion)
            StabilityList.append(static_stability)
            TemperatureList.append(temperature)
            VirtualList.append(virtual_temperature)
            HumidityList.append(relative_humidity)
            ThicknessList.append(pressure_thickness)
            WaterList.append(precipitable_water)

            # Randomise the initial conditions.
            randomise = (2 * random_sample(np.shape(geo_i)) - 1)
            randomise /= 4
            # Geopotential Height.
            geo_rand = randomise * geo_dev
            geo_i = geo_initial + geo_rand
            # Temperature.
            T_rand = randomise * T_dev
            T_i = T_initial + T_rand
            # Virtual Temperature.
            Tv_rand = randomise * Tv_dev
            Tv_i = Tv_initial + Tv_rand

            # Redefine the otherr output variables in
            # accordance with these newly defined initial
            # conditions.
            # Geopotential Height.
            height = geo_i / g
            # Relative Humidity.
            # Vapor pressure.
            e = (
                (
                    (pressure_3d) / (1 - 0.622)
                ) - (
                    (pressure_3d * T_i) / (Tv_i * (1 - 0.622))
                )
            )
            # Convert temperature in Kelvin to degrees centigrade.
            T_c = T_i.value - 273.15
            # Saturated vapor pressure.
            sat_vapor_pressure = 0.61121 * np.exp(
                (
                    18.678 - (T_c / 234.5)
                ) * (T_c / (257.14 + T_c)
                )
            ) 
            # Add units of measurement.
            sat_vapor_pressure *= self.units.kPa
            sat_vapor_pressure = sat_vapor_pressure.to(self.units.hPa)
            # Determine relative humidity.
            rh = (e.value / sat_vapor_pressure.value) * 100
            rh[rh > 100] = 100
            rh[rh < 0] = 0
            rh *= self.units.percent
            # Configure the Wind class, so, that it aligns with the
            # paramaters defined by the user.
            config = Wind(
                delta_latitude=self.delta_latitude,
                delta_longitude=self.delta_longitude,
                remove_files=self.remove_files,
                input_data=True, 
                geo=height, 
                temp=T_i, 
                rh=rh, 
            )
            # Geostrophic Wind.
            # Zonal Wind.
            u_g = config.zonal_wind()
            # Meridional Wind.
            v_g = config.meridional_wind()
            # Vertical Motion.
            omega = config.vertical_motion()
            # Static Stability.
            sigma = config.static_stability()
            # Precipitable Water.
            pwv = config.precipitable_water()

            m += 1

        # Convert lists to arrays, and generate the mean forecast.
        # Geopotential Height.
        GeopotentialHeight = np.asarray(HeightList)
        GeopotentialHeight[:, 0] = self.geopotential_height()
        # Zonal Wind.
        ZonalWind = np.asarray(ZonalList)
        ZonalWind[:, 0] = self.zonal_wind()
        # Meridional Wind.
        MeridionalWind = np.asarray(MeridionalList)
        MeridionalWind[:, 0] = self.meridional_wind()
        # Vertical Motion.
        VerticalMotion = np.asarray(VerticalList)
        VerticalMotion[:, 0] = self.vertical_motion()
        # Static Stability.
        StaticStability = np.asarray(StabilityList)
        StaticStability[:, 0] = self.static_stability()
        # Air Temperature.
        AirTemperature = np.asarray(TemperatureList)
        AirTemperature[:, 0] = self.temperature()
        # Virtual Temperature.
        VirtualTemperature = np.asarray(VirtualList)
        VirtualTemperature[:, 0] = self.virtual_temperature()
        # Relative Humidity.
        RelativeHumidity = np.asarray(HumidityList)
        RelativeHumidity[:, 0] = self.relative_humidity()
        # Pressure Thickness.
        Thickness = np.asarray(ThicknessList)
        Thickness[:, 0] = self.pressure_thickness(p1=p1, p2=p2)
        # Precipitable Water.
        PrecipitableWater = np.asarray(WaterList)
        PrecipitableWater[:, 0] = self.precipitable_water()

        # Finish progress bar.
        bar.finish()

        # Define the coordinates for the cube. 
        # Latitude.
        lat = DimCoord(
            self.latitude_lines(),
            standard_name='latitude',
            units='degrees'
        )
        # Longitude
        lon = DimCoord(
            self.longitude_lines(),
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
        # Model number.
        model_runs = np.linspace(1, self.models, self.models + 1)
        model_runs = DimCoord(
            model_runs,
            standard_name='efs_model', 
        )

        # Cubes
        # Forecast reference time.
        ref_time = AuxCoord(
            self.date.strftime("%Y-%m-%d %H:%M:%S"),
            standard_name='forecast_reference_time'
        )
        # Geopotential Height Cube.
        height_cube = Cube(GeopotentialHeight,
            standard_name='geopotential_height',
            units='m',
            dim_coords_and_dims=[
                (model_runs, 0), (forecast_period, 1), (p, 2), (lat, 3), (lon, 4)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        height_cube.add_aux_coord(ref_time)
        # Geostrophic Wind Cubes.
        # Zonal Wind Cube.
        u_cube = Cube(ZonalWind,
            standard_name='x_wind',
            units='m s-1',
            dim_coords_and_dims=[
                (model_runs, 0), (forecast_period, 1), (p, 2), (lat, 3), (lon, 4)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        u_cube.add_aux_coord(ref_time)
        # Meridional Wind Cube.
        v_cube = Cube(MeridionalWind,
            standard_name='y_wind',
            units='m s-1',
            dim_coords_and_dims=[
                (model_runs, 0), (forecast_period, 1), (p, 2), (lat, 3), (lon, 4)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        v_cube.add_aux_coord(ref_time)
        # Vertical Motion.
        omega_cube = Cube(VerticalMotion,
            standard_name='lagrangian_tendency_of_air_pressure',
            units='hPa s-1',
            dim_coords_and_dims=[
                (model_runs, 0), (forecast_period, 1), (p, 2), (lat, 3), (lon, 4)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        omega_cube.add_aux_coord(ref_time)
        # Static Stability.
        sigma_cube = Cube(StaticStability,
            long_name='static_stability',
            units='J hPa-2 kg-1',
            dim_coords_and_dims=[
                (model_runs, 0), (forecast_period, 1), (p, 2), (lat, 3), (lon, 4)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        sigma_cube.add_aux_coord(ref_time)
        # Temperature.
        # Air Temperature.
        T_cube = Cube(AirTemperature,
            standard_name='air_temperature',
            units='K',
            dim_coords_and_dims=[
                (model_runs, 0), (forecast_period, 1), (p, 2), (lat, 3), (lon, 4)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        T_cube.add_aux_coord(ref_time)
        # Virtual Temperature.
        Tv_cube = Cube(VirtualTemperature,
            standard_name='virtual_temperature',
            units='K',
            dim_coords_and_dims=[
                (model_runs, 0), (forecast_period, 1), (p, 2), (lat, 3), (lon, 4)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        Tv_cube.add_aux_coord(ref_time)
        # Relative Humidity.
        rh_cube = Cube(RelativeHumidity,
            standard_name='relative_humidity',
            units='%',
            dim_coords_and_dims=[
                (model_runs, 0), (forecast_period, 1), (p, 2), (lat, 3), (lon, 4)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        rh_cube.add_aux_coord(ref_time)
        # Pressure Thickness.
        thickness_cube = Cube(Thickness,
            long_name='pressure_thickness',
            units='m',
            dim_coords_and_dims=[
                (forecast_period, 0), (lat, 1), (lon, 2)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        thickness_cube.add_aux_coord(ref_time)
        # Precipitable Water.
        pwv_cube = Cube(PrecipitableWater,
            long_name='precipitable_water',
            units='mm',
            dim_coords_and_dims=[
                (forecast_period, 0), (lat, 1), (lon, 2)
            ],
            attributes={
                'source': 'Motus Aeris @ AMSIMP',
            }
        )
        pwv_cube.add_aux_coord(ref_time)

        # Create Cube list of output parameters.
        output = CubeList([
            height_cube,
            u_cube,
            v_cube,
            omega_cube,
            sigma_cube,
            T_cube,
            Tv_cube,
            rh_cube,
            thickness_cube,
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

        return output

    def simulate(self):
        """
        This method outputs a visualisation of how temperature, pressure
        thickness, atmospheric pressure, and precipitable water vapor will 
        evolve. The atmospheric pressure and precipitable water elements of
        this visualisation operate similarly to the method, 
        amsimp.Backend.longitude_contourf(), so, please refer to this method for
        a detailed description of the aforementioned elements. Likewise, please
        refer to amsimp.Water.water_contourf() for more information on the
        visualisation element of precipitable water vapor, or to
        amsimp.Backend.altitude_contourf() for more information on the
        visualisation element of temperature.

        For a visualisation of geostrophic wind (zonal and meridional
        components), please refer to the amsimp.Wind.wind_contourf(), or
        amsimp.Wind.globe() methods.
        """
        # Define the forecast period.
        forecast_days = int(self.forecast_days)
        cdef np.ndarray time = np.linspace(
            0, forecast_days, (forecast_days * 24)
        )

        # Style of graph.
        style.use("fivethirtyeight")

        # Define layout.
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(18.5, 7.5))
        fig.subplots_adjust(hspace=0.340, bottom=0.105, top=0.905)
        plt.ion()

        # Temperature
        indx_long = (np.abs(self.longitude_lines().value - 0)).argmin()
        ax1 = plt.subplot(gs[0, 0])
        forecast_temp = self.atmospheric_prognostic_method()
        # Atmospheric Pressure
        ax2 = plt.subplot(gs[1, 0], projection=ccrs.EckertIII())
        forecast_pressure = self.atmospheric_prognostic_method()
        # Precipitable Water
        ax3 = plt.subplot(gs[0, 1], projection=ccrs.EckertIII())
        forecast_Pwv = self.atmospheric_prognostic_method()
        # Pressure Thickness (1000hPa - 500hPa).
        ax4 = plt.subplot(gs[1, 1], projection=ccrs.EckertIII())
        forecast_pthickness = self.atmospheric_prognostic_method()

        # Troposphere - Stratosphere Boundary Line
        trop_strat_line = self.troposphere_boundaryline()
        trop_strat_line = (
            np.zeros(len(trop_strat_line.value)) + np.mean(trop_strat_line)
        )

        t = 0
        while t < len(time):
            # Defines the axes.
            # For the temperature contour plot.
            latitude, altitude = np.meshgrid(
                self.latitude_lines(), self.altitude_level()
            )
            # For the pressure, precipitable water, and pressure 
            # thickness countour plot
            lat, long = np.meshgrid(
                self.latitude_lines(), self.longitude_lines()
            )
            
            # Temperature contour plot.
            # Temperature data.
            temperature = forecast_temp[t]
            temperature = temperature[indx_long, :, :]
            temperature = np.transpose(temperature)

            # Contour plotting.
            cmap1 = plt.get_cmap("hot")
            min1 = np.min(forecast_temp)
            max1 = np.max(forecast_temp)
            level1 = np.linspace(min1, max1, 21)
            temp = ax1.contourf(
                latitude, altitude, temperature, cmap=cmap1, levels=level1
            )

            # Checks for a colorbar.
            if t == 0:
                cb1 = fig.colorbar(temp, ax=ax1)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb1.locator = tick_locator
                cb1.update_ticks()
                cb1.set_label("Temperature (K)")

            # Add SALT to the graph.
            ax1.set_xlabel("Latitude ($\phi$)")
            ax1.set_ylabel("Altitude (m)")
            ax1.set_title("Temperature")

            # Atmospheric pressure contour.
            # Pressure data.
            pressure = forecast_pressure[t]
            pressure = pressure[:, :, 0]

            # EckertIII projection details.
            ax2.set_global()
            ax2.coastlines()
            ax2.gridlines()

            # Contourf plotting.
            pressure_sealevel = np.asarray(forecast_pressure)
            pressure_sealevel = pressure_sealevel[:, :, :, 0]
            cmap2 = plt.get_cmap("jet")
            min2 = np.min(pressure_sealevel)
            max2 = np.max(pressure_sealevel)
            level2 = np.linspace(min2, max2, 21)
            atmospheric_pressure = ax2.contourf(
                long,
                lat,
                pressure,
                cmap=cmap2,
                levels=level2,
                transform=ccrs.PlateCarree(),
            )

            # Checks for a colorbar.
            if t == 0:
                cb2 = fig.colorbar(atmospheric_pressure, ax=ax2)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb2.locator = tick_locator
                cb2.update_ticks()
                cb2.set_label("Pressure (hPa)")

            # Add SALT to the graph.
            ax2.set_xlabel("Longitude ($\lambda$)")
            ax2.set_ylabel("Latitude ($\phi$)")
            ax2.set_title(
                "Atmospheric Pressure (Alt = " 
                + str(self.altitude_level()[0])
                + ")"
            )

            # Precipitable water contour.
            # Precipitable water data.
            precipitable_water = forecast_Pwv[t]

            # EckertIII projection details.
            ax3.set_global()
            ax3.coastlines()
            ax3.gridlines()

            # Contourf plotting.
            cmap3 = plt.get_cmap("seismic")
            min3 = np.min(forecast_Pwv)
            max3 = np.max(forecast_Pwv)
            level3 = np.linspace(min3, max3, 21)
            precipitable_watervapour = ax3.contourf(
                long,
                lat,
                precipitable_water,
                cmap=cmap3,
                levels=level3,
                transform=ccrs.PlateCarree(),
            )

            # Checks for a colorbar.
            if t == 0:
                cb3 = fig.colorbar(precipitable_watervapour, ax=ax3)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb3.locator = tick_locator
                cb3.update_ticks()
                cb3.set_label("Precipitable Water (mm)")

            # Add SALT to the graph.
            ax3.set_xlabel("Longitude ($\lambda$)")
            ax3.set_ylabel("Latitude ($\phi$)")
            ax3.set_title("Precipitable Water")

            # Pressure thickness scatter plot.
            # Pressure thickness data.
            pressure_thickness = forecast_pthickness[t]

            # EckertIII projection details.
            ax4.set_global()
            ax4.coastlines()
            ax4.gridlines()

            # Contourf plotting.
            min4 = np.min(forecast_pthickness)
            max4 = np.max(forecast_pthickness)
            level4 = np.linspace(min4, max4, 21)
            pressure_h = ax4.contourf(
                long,
                lat,
                pressure_thickness,
                transform=ccrs.PlateCarree(),
                levels=level4,
            )

            # Index of the rain / snow line
            indx_snowline = (np.abs(level4 - 5400)).argmin()
            pressure_h.collections[indx_snowline].set_color('black')
            pressure_h.collections[indx_snowline].set_linewidth(1) 

            # Add SALT.
            ax4.set_xlabel("Latitude ($\phi$)")
            ax4.set_ylabel("Longitude ($\lambda$)")
            ax4.set_title("Thickness (1000 hPa - 500 hPa)")

            # Checks for a colorbar.
            if t == 0:
                cb4 = fig.colorbar(pressure_h, ax=ax4)
                tick_locator = ticker.MaxNLocator(nbins=10)
                cb4.locator = tick_locator
                cb4.update_ticks()
                cb4.set_label("Pressure Thickness (m)")

            # Plots the average boundary line on two contourfs.
            # Temperature contourf.
            ax1.plot(
                latitude[1],
                trop_strat_line,
                color="black",
                linestyle="dashed",
                label="Troposphere - Stratosphere Boundary Line",
            )
            ax1.legend(loc=0)

            # Title of Simulation.
            now = self.date + timedelta(hours=+t)
            hour = now.hour
            minute = now.minute

            if minute < 10:
                minute = "0" + str(minute)
            else:
                minute = str(minute)

            fig.suptitle(
                "Motus Aeris @ AMSIMP (" + str(hour)
                + ":" + minute + " "
                + now.strftime("%d-%m-%Y") + ")"
            )

            # Displaying simualtion.
            plt.show()
            plt.pause(0.01)
            if t < (len(time) - 1):
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
            else:
                plt.pause(10)

            note = (
                "Note: Geostrophic balance does not hold near the equator."
                + " Rain / Snow Line is marked on the Pressure Thickness" 
                + " contour plot by the black line (5,400 m)."
            )

            # Footnote
            plt.figtext(
                0.99,
                0.01,
                note,
                horizontalalignment="right",
                fontsize=10,
            )

            t += 1
