"""
AMSIMP Operational Model Class. For information about this
class is described below.

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

# Importing Dependencies.
import os
import sys
from tqdm import tqdm
import iris
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dropout, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from amsimp.preprocessing import Preprocessing

# -----------------------------------------------------------------------------------------#


class OperationalModel(Preprocessing):
    """
    This is the operational model class for AMSIMP.
    """

    def model_architecture(self):
        r"""Generates the operational AMSIMP Global Forecast Model architecture.

        Returns
        -------
        `tf.keras.Sequential`
            Operational AMSIMP Global Forecast Model architecture.

        Notes
        -----
        This architecture is currently based on the ConvLSTM layer, which has
        been pretrained on the dataset from the year 2009 to the year 2016.
        A major drawback of LSTMs in its handling of spatiotemporal data is due
        to its usage of full connections in input-to-state and state-to-state
        transitions in which no spatial information is encoded. To overcome
        this problem, a distinguishing feature of a ConvLSTM cell is that all
        the inputs and gates of the ConvLSTM layer are 3D tensors whose last
        two dimensions are spatial dimensions.
        """
        # Create, and train models.
        # Optimiser.
        opt = Adam(lr=1e-3, decay=1e-5)
        # Create model.
        model = Sequential()

        # First layer.
        model.add(
            ConvLSTM2D(
                filters=64,
                kernel_size=(7, 7),
                input_shape=(6, 179, 360, 3),
                padding="same",
                return_sequences=True,
                activation="tanh",
                recurrent_activation="hard_sigmoid",
                kernel_initializer="glorot_uniform",
                unit_forget_bias=True,
                dropout=0.3,
                recurrent_dropout=0.3,
                go_backwards=True,
            )
        )
        # Batch normalisation.
        model.add(BatchNormalization())
        # Dropout.
        model.add(Dropout(0.1))

        # Second layer.
        model.add(
            ConvLSTM2D(
                filters=32,
                kernel_size=(7, 7),
                padding="same",
                return_sequences=True,
                activation="tanh",
                recurrent_activation="hard_sigmoid",
                kernel_initializer="glorot_uniform",
                unit_forget_bias=True,
                dropout=0.4,
                recurrent_dropout=0.3,
                go_backwards=True,
            )
        )
        # Batch normalisation.
        model.add(BatchNormalization())

        # Third layer.
        model.add(
            ConvLSTM2D(
                filters=32,
                kernel_size=(7, 7),
                padding="same",
                return_sequences=True,
                activation="tanh",
                recurrent_activation="hard_sigmoid",
                kernel_initializer="glorot_uniform",
                unit_forget_bias=True,
                dropout=0.4,
                recurrent_dropout=0.3,
                go_backwards=True,
            )
        )
        # Batch normalisation.
        model.add(BatchNormalization())
        # Dropout.
        model.add(Dropout(0.1))

        # Final layer.
        model.add(
            ConvLSTM2D(
                filters=32,
                kernel_size=(7, 7),
                padding="same",
                return_sequences=True,
                activation="tanh",
                recurrent_activation="hard_sigmoid",
                kernel_initializer="glorot_uniform",
                unit_forget_bias=True,
                dropout=0.5,
                recurrent_dropout=0.3,
                go_backwards=True,
            )
        )
        # Batch normalisation.
        model.add(BatchNormalization())

        # Add dense layer.
        model.add(Dense(3))

        # Compile model.
        model.compile(optimizer=opt, loss="mse", metrics=["mean_absolute_error"])

        return model

    def generate_forecast(self):
        r"""Generates a forecast with the current AMSIMP Global Forecast Model
        architecture.

        Returns
        -------
        `iris.cube.CubeList`
            The forecast generated with the operational model

        Notes
        -----
        This model has been pretrained on the dataset from the year 2009 to the
        year 2016. The architecture of the current operational model is
        currently based on the ConvLSTM layer. The prognostic variables are: air
        temperature at 2 metres above the surface, air temperature at a pressure
        surface of 850 hectopascals, and geopotential at a pressure surface of
        500 hectopascals.

        See Also
        --------
        normalise_dataset
        """
        # Define model input.
        model_input = self.normalise_dataset()

        # Define directory.
        import amsimp.preprocessing

        directory = os.path.dirname(amsimp.preprocessing.__file__)

        # Define model and load weights.
        model = self.model_architecture()
        model.load_weights(directory + "/model/global_forecast_model.h5")

        # Define forecast output array.
        model_output = np.zeros(
            (
                int((self.forecast_length.value / 2) + 1),
                model_input.shape[2],
                model_input.shape[3],
                model_input.shape[4],
            )
        )
        # Add current conditions to output array.
        model_output[0] = model_input[0, -1]

        # Create iterative predictions.
        it = 1
        pbar = tqdm(
            total=int(self.forecast_length.value / 12), desc="Generating forecast"
        )
        while it < (self.forecast_length.value / 2):
            # Generate predictions based on current model input.
            predictions = model.predict(model_input)

            # Add predictions to output array.
            model_output[it : it + 6] = predictions[0]

            # Define as new model input.
            model_input = predictions

            # Increment iteration.
            it += 6
            pbar.update()

        # Close progress bar.
        pbar.close()

        # Define progress bar.
        pbar = tqdm(total=3, desc="Outputting forecast")

        # Inverse of normalisation.
        # Load normalisation variables.
        # Mean.
        mean = np.load(directory + "/model/mean.npy")
        # Standard deviation.
        std = np.load(directory + "/model/std.npy")

        # Determine inverse.
        model_output = (model_output * std) + mean

        # Define forecast for each parameter.
        # 2 metre temperature.
        t2m = model_output[:, :, :, 0]

        # 850 hPa temperature.
        t = model_output[:, :, :, 1]

        # 500 hPa geopotential.
        z = model_output[:, :, :, 2]

        # Define time coordinate.
        time = self.interpolate_dataset()[0].coord("time")
        # Define unit of measurement.
        time_unit = time.units
        # Define values.
        time = (
            np.linspace(0, self.forecast_length.value, model_output.shape[0])
            + time.points[-1]
        )

        # Define the coordinates for the cubes.
        # Time.
        time = iris.coords.DimCoord(time, standard_name="time", units=time_unit)
        # Latitude.
        lat = iris.coords.DimCoord(
            self.lat(), standard_name="latitude", units="degrees"
        )
        # Longitude
        lon = iris.coords.DimCoord(
            self.lon(), standard_name="longitude", units="degrees"
        )
        #  Aux coords.
        # 850 hPa for temperature.
        p850 = iris.coords.AuxCoord(
            np.array([850]), standard_name="air_pressure", units="hPa"
        )
        # 500 hPa for geopotential.
        p500 = iris.coords.AuxCoord(
            np.array([500]), standard_name="air_pressure", units="hPa"
        )

        # Define cubes.
        # 2 metre temperature.
        t2m = iris.cube.Cube(
            t2m,
            long_name="2m_temperature",
            var_name="t2m",
            units="K",
            dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
            attributes={
                "source": "AMSIMP Global Forecast Model",
            },
        )
        pbar.update()

        # 850 hPa temperature.
        t = iris.cube.Cube(
            t,
            standard_name="air_temperature",
            var_name="t",
            units="K",
            dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
            attributes={
                "source": "AMSIMP Global Forecast Model",
            },
        )
        t.add_aux_coord(p850)
        pbar.update()

        # 500 hPa geopotential.
        z = iris.cube.Cube(
            z,
            standard_name="geopotential",
            var_name="z",
            units="m2 s-2",
            dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
            attributes={
                "source": "AMSIMP Global Forecast Model",
            },
        )
        z.add_aux_coord(p500)
        pbar.update()

        # Finish progress bar.
        pbar.close()

        # Define output forecast.
        forecast = iris.cube.CubeList([t2m, t, z])

        return forecast
