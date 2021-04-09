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
import tensorflow as tf
from tqdm import tqdm
import iris
import numpy as np
from amsimp.preprocessing import Preprocessing

# -----------------------------------------------------------------------------------------#


class OperationalModel(Preprocessing):
    """
    This is the operational model class for AMSIMP.
    """

    def generate_forecast(self, save=False):
        r"""Generates a forecast with the current AMSIMP Global Forecast Model
        architecture.

        Returns
        -------
        `iris.cube.CubeList`
            The forecast generated with the operational model

        Notes
        -----
        This model has been pretrained on the dataset from the year 1980 to the
        year 2018. The architecture of the current operational model is
        currently based on the ConvLSTM layer. The prognostic variables are: air
        temperature and geopotential at various pressure surfaces.

        See Also
        --------
        load_models
        """
        # Define model input.
        model_input = self.normalise_dataset()

        # Define forecast output array.
        model_output = np.zeros(
            (
                int((self.forecast_length.value / 2) + 1),
                model_input.shape[0],
                model_input.shape[1],
                model_input.shape[2],
                model_input.shape[3],
            )
        )
        # Add current conditions to output array.
        model_output[0] = model_input

        # Define directory.
        import amsimp.preprocessing

        directory = os.path.dirname(amsimp.preprocessing.__file__)

        # Define model.
        model = tf.keras.models.load_model(directory + "/gfm/model")

        # Create a progress bar.
        pbar = tqdm(
            total=int(self.forecast_length.value / 2),
            desc="Generating forecast",
        )

        # Create iterative predictions.
        it = 2
        while it <= self.forecast_length.value:
            # Generate predictions based on current model input.
            prediction = model.predict(model_input)

            # Add predictions to output array.
            model_output[int(it / 2)] = prediction

            # Define as new model input.
            model_input = prediction

            # Increment iteration.
            it += 2

            # Increment for progress bar.
            pbar.update()
            
        # Complete progress bar.
        pbar.close()

        # Load normalisation variables.
        # Mean.
        mean = np.load(directory + "/gfm/mean.npy")

        # Standard deviation.
        std = np.load(directory + "/gfm/std.npy")

        # Inverse of normalisation.
        model_output = (model_output * std) + mean

        # Define progress bar.
        pbar = tqdm(total=2, desc="Outputting forecast")

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

        # Forecast types.
        if self.use_efs:
            # Ensemble forecast system.
            # Air temperature.
            t = model_output[:, :, :, :, 0]

            # Geopotential.
            z = model_output[:, :, :, :, 1]

            # Ensemble member dimcoord.
            realization = iris.coords.DimCoord(
                np.linspace(0, 30, 31), standard_name="realization"
            )

            # Co-ordinates.
            coords = [(time, 0), (realization, 1), (lat, 2), (lon, 3)]

            #  Source label.
            source = "AMSIMP Global Ensemble Forecast Model"
        else:
            # Deterministic forecast system.
            # Air temperature.
            t = model_output[:, 0, :, :, 0]

            # Geopotential.
            z = model_output[:, 0, :, :, 1]

            # Co-ordinates.
            coords = [(time, 0), (lat, 1), (lon, 2)]

            #  Source label.
            source = "AMSIMP Global Deterministic Forecast Model"

        # Define cubes.
        # Air temperature.
        t = iris.cube.Cube(
            t,
            standard_name="air_temperature",
            var_name="t",
            units="K",
            dim_coords_and_dims=coords,
            attributes={
                "source": source,
            },
        )
        pbar.update()

        # Geopotential.
        z = iris.cube.Cube(
            z,
            standard_name="geopotential",
            var_name="z",
            units="m2 s-2",
            dim_coords_and_dims=coords,
            attributes={
                "source": source,
            },
        )
        pbar.update()

        # Finish progress bar.
        pbar.close()

        # Define output forecast.
        forecast = iris.cube.CubeList([t, z])

        # Initialism.
        if self.use_efs:
            # Model initialism.
            initialism = "gefm"
        else:
            # Model initialism.
            initialism = "gfm"

        # Save forecast if requested.
        if save:
            iris.save(
                forecast,
                "amsimp_{}_{}.nc".format(
                    initialism, time.units.num2date(time.points[0]).strftime("%Y%m%d%H")
                ),
            )

        return forecast
