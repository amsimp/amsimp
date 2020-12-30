"""
AMSIMP Developmental and Operational Forecasting Classes. For information
about these classes is described below.

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
import tensorflow as tf
from metpy.calc import smooth_gaussian
from amsimp.preprocessing import PreProcessing

# -----------------------------------------------------------------------------------------#


class _DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, lst_n, batch_size=32, shuffle=True):
        """
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        """
        # Define variables.
        self.lst_n = lst_n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.folder = "processed_dataset/"
        self.fname = "sample"
        self.shape = (self.batch_size, 6, 721, 1440, 4)

        # Extra variables.
        self.n_samples = lst_n.shape[0]
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        "Generate one batch of data"
        idxs = self.idxs[i * self.batch_size : (i + 1) * self.batch_size]

        # Define outputs.
        X = np.zeros(self.shape)
        y = np.zeros(self.shape)

        # Get file names and load.
        n = 0
        for k in idxs:
            # Define file name.
            fname = self.fname + str(int(self.lst_n[k])) + ".npy"

            # Load file.
            file = np.load(self.folder + fname)

            # Reduce spatial resolution.
            X[n] = file[0]
            y[n] = file[1]

            # Increment.
            n += 1

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


class DevelopmentalModel:
    """
    Explain here.
    """

    def __init__(
        self, raw_dataset=None, timesteps=2, processed_dataset=None, cal_vars=False
    ):
        """
        Explain here.
        """
        # Make the aforementioned variables available else where in the class.
        self.raw_dataset = raw_dataset
        self.timesteps = timesteps
        self.processed_dataset = processed_dataset
        self.cal_vars = cal_vars

        # Check if directory provided to the software exists.
        if not os.path.exists(self.raw_dataset):
            raise FileNotFoundError(
                "The raw dataset directory provided could not be located."
            )

        # Ensure self.timesteps is greater than, or equal to 1.
        if self.timesteps.value <= 0:
            raise ValueError(
                "The parameter, timesteps, must be a positive number greater than, or equal to 1. "
                + "The value of timesteps was: {}".format(self.timesteps)
            )

        # Ensure self.cal_vars is a boolean value.
        if not isinstance(self.cal_vars, bool):
            raise ValueError("The parameter, cal_vars, must be a boolean value.")

        # Check if directory provided to the software exists.
        if not os.path.exists(self.processed_dataset):
            raise FileNotFoundError(
                "The raw dataset directory provided could not be located."
            )

    def __window_data(self, dataset, label, past_history, future_target):
        """
        Explain here.
        """
        # Define output lists.
        X, y = [], []

        # Set description for progress bar.
        desc = "Windowing {} dataset".format(label)

        # Loop through dataset.
        for i in tqdm(range(dataset.shape[0]), desc=desc):
            # Find the end.
            end_ix = i + past_history
            out_end_ix = end_ix + future_target

            # Determine if we are beyond the dataset.
            if out_end_ix > dataset.shape[0]:
                break

            # Gather the input and output components.
            seq_x = dataset[i:end_ix]
            seq_y = dataset[end_ix:out_end_ix]

            # Append to list.
            X.append(seq_x)
            y.append(seq_y)

        return X, y

    def load_training_dataset(self):
        """
        Explain here.
        """
        # Load dataset.
        dataset = iris.load(self.raw_dataset + "/*/*.nc")
        var_list = ["t2m", "tp", "t", "z"]

        # Extract the correct variables.
        dataset = dataset.extract(var_list)

        # Ensure all parameters are present.
        if len(dataset) != 4:
            raise Exception(
                "All of the expected parameters were not present in the dataset."
            )

        return dataset

    def normalisation_variables(self):
        """
        Explain here.
        """
        # Define dataset.
        dataset = self.load_training_dataset()

        # Determine.
        mean = np.zeros((len(dataset)))
        std = np.zeros_like(mean)

        # Define progress bar.
        t = tqdm(total=len(dataset) * 2, desc="Calculating normalisation variables")

        # Loop.
        for i in range(len(dataset)):
            # Mean.
            var_mean = (
                dataset[i]
                .collapsed(["time", "latitude", "longitude"], iris.analysis.MEAN)
                .data
            )
            t.update()

            # Standard deviation.
            var_std = (
                dataset[i]
                .collapsed("time", iris.analysis.STD_DEV)
                .collapsed(["latitude", "longitude"], iris.analysis.MEAN)
                .data
            )
            t.update()

            # Append  to list.
            mean[i] = var_mean
            std[i] = var_std

        # Close progress bar.
        t.close()

        # Define directory.
        import amsimp.preprocessing

        directory = os.path.dirname(amsimp.preprocessing.__file__)

        # Save to array.
        np.save(directory + "/model/mean.npy", mean)
        np.save(directory + "/model/std.npy", std)

        return mean, std

    def preprocess_training_dataset(self):
        """
        Explain here.
        """
        # Define dataset.
        dataset = self.load_training_dataset()

        # Load dataset.
        # 2 metre temperature.
        temperature = dataset[0]

        # Total precipitation.
        total_precipitation = dataset[1]

        # 850 hPa Temperature.
        temperature_850 = dataset[2]

        # 500 hPa Geopotential.
        geopotential = dataset[3]

        # Window datasets.
        # 2 metre temperature.
        windowed_t2m = self.__window_data(
            temperature, "2 metre temperature", self.timesteps, self.timesteps
        )

        # Total precipitation.
        windowed_tp = self.__window_data(
            total_precipitation, "Total precipitation", self.timesteps, self.timesteps
        )

        # 850 hPa Temperature.
        windowed_t = self.__window_data(
            temperature_850, "850 hPa temperature", self.timesteps, self.timesteps
        )

        # 500 hPa Geopotential.
        windowed_z = self.__window_data(
            geopotential, "500 hPa geopotential", self.timesteps, self.timesteps
        )

        # Number of samples.
        n_samples = len(windowed_t2m[0])
        assert n_samples == len(windowed_t2m[1])

        # Loop through windowed dataset, convert to NumPy arrays, normalise
        # dataset, and save as individual files.
        # Define variables.
        if self.cal_vars:
            # Calculate based on provided dataset.
            mean, std = self.normalisation_variables()
        else:
            # Define directory.
            import amsimp.preprocessing

            directory = os.path.dirname(amsimp.preprocessing.__file__)

            # Load existing normalisation variables.
            # Load normalisation variables.
            # Mean.
            mean = np.load(directory + "/model/mean.npy")

            # Standard deviation.
            std = np.load(directory + "/model/std.npy")

        for it in tqdm(range(n_samples), desc="Preprocessing training dataset"):
            # Select each window for each parameter and convert to NumPy array.
            # 2 metre temperature.
            X_t2m, y_t2m = windowed_t2m[0][it].copy(), windowed_t2m[1][it].copy()
            X_t2m, y_t2m = X_t2m.data, y_t2m.data

            # Total precipitation.
            X_tp, y_tp = windowed_tp[0][it].copy(), windowed_tp[1][it].copy()
            X_tp, y_tp = X_tp.data, y_tp.data

            # 850 hPa Temperature.
            X_t, y_t = windowed_t[0][it].copy(), windowed_t[1][it].copy()
            X_t, y_t = X_t.data, y_t.data

            # 500 hPa Geopotential.
            X_z, y_z = windowed_z[0][it].copy(), windowed_z[1][it].copy()
            X_z, y_z = X_z.data, y_z.data

            # Combine each parameter into single array.
            X = np.array([X_t2m, X_tp, X_t, X_z])
            y = np.array([y_t2m, y_tp, y_t, y_z])

            # Transpose array to output format.
            X = np.transpose(X, (1, 2, 3, 0))
            y = np.transpose(y, (1, 2, 3, 0))

            # Normalise.
            X = (X - mean) / std
            y = (y - mean) / std

            # Save.
            sample = np.array([X, y])
            np.save(
                self.processed_dataset + "/sample" + str(it + 1) + ".npy", sample
            )

    def global_forecast_model(self):
        """
        Explain here.
        """
        # Import solely in the event the model architecture is required.
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import ConvLSTM2D, Dropout, Dense
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.mixed_precision import experimental as mixed_precision

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
                padding='same', 
                return_sequences=True, 
                activation='tanh', 
                recurrent_activation='hard_sigmoid',
                kernel_initializer='glorot_uniform', 
                unit_forget_bias=True, 
                dropout=0.3, 
                recurrent_dropout=0.3, 
                go_backwards=True
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
                padding='same', 
                return_sequences=True, 
                activation='tanh', 
                recurrent_activation='hard_sigmoid', 
                kernel_initializer='glorot_uniform', 
                unit_forget_bias=True, 
                dropout=0.4, 
                recurrent_dropout=0.3, 
                go_backwards=True
            )
        )
        # Batch normalisation.
        model.add(BatchNormalization())
        
        # Third layer.
        model.add(
            ConvLSTM2D(
                filters=32, 
                kernel_size=(7, 7), 
                padding='same', 
                return_sequences=True, 
                activation='tanh', 
                recurrent_activation='hard_sigmoid', 
                kernel_initializer='glorot_uniform', 
                unit_forget_bias=True, 
                dropout=0.4, 
                recurrent_dropout=0.3, 
                go_backwards=True
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
                padding='same', 
                return_sequences=True, 
                activation='tanh', 
                recurrent_activation='hard_sigmoid', 
                kernel_initializer='glorot_uniform', 
                unit_forget_bias=True, 
                dropout=0.5, 
                recurrent_dropout=0.3, 
                go_backwards=True
            )
        )
        # Batch normalisation.
        model.add(BatchNormalization())

        # Add dense layer.
        model.add(Dense(3))
    
        # Compile model.
        model.compile(
            optimizer=opt, 
            loss='mse', 
            metrics=['mean_absolute_error']
        )

        return model

    def train_model(self, model, epochs=50, bs=4):
        """
        Explain here.
        """
        # Number of elements.
        n = len(os.listdir(self.processed_dataset))
        lst_n = np.linspace(1, n, n)

        # Define training dataset and validation dataset.
        # Training.
        train_ns = lst_n[: int(0.7 * n)]
        train_generator = _DataGenerator(
            train_ns,
            batch_size=bs,
        )

        # Validation.
        val_ns = lst_n[int(0.7 * n) : int(0.9 * n)]
        val_generator = _DataGenerator(val_ns, batch_size=bs, shuffle=False)

        # Train.
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", min_delta=0, patience=2, mode="auto"
                )
            ],
        )

        return model


# -----------------------------------------------------------------------------------------#


class OperationalModel(PreProcessing, DevelopmentalModel):
    """
    Explain here.
    """
    
    def generate_forecast(self):
        """
        Explain here.
        """
        # Define model input.
        model_input = self.normalise_dataset()

        # Define directory.
        import amsimp.preprocessing
        directory = os.path.dirname(amsimp.preprocessing.__file__)

        # Define model and load weights.
        model = self.global_forecast_model()
        model.load_weights(directory + '/model/global_forecast_model.h5')

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
        pbar = tqdm(total=int(self.forecast_length.value / 12), desc='Generating forecast')
        while it < (self.forecast_length.value / 2):
            # Generate predictions based on current model input.
            predictions = model.predict(model_input)

            # Add predictions to output array.
            model_output[it:it+6] = predictions[0]

            # Define as new model input.
            model_input = predictions

            # Increment iteration.
            it += 6
            pbar.update()
        
        # Close progress bar.
        pbar.close()
        
        # Define progress bar.
        pbar = tqdm(total=3, desc='Outputting forecast')

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
        time = self.interpolate_dataset()[0].coord('time')
        # Define unit of measurement.
        time_unit = time.units
        # Define values.
        time = np.linspace(
            0, self.forecast_length.value, model_output.shape[0]
        ) + time.points[-1]

        # Define the coordinates for the cubes. 
        # Time.
        time = iris.coords.DimCoord(
            time,
            standard_name='time', 
            units=time_unit
        )
        # Latitude.
        lat = iris.coords.DimCoord(
            self.lat(),
            standard_name='latitude',
            units='degrees'
        )
        # Longitude
        lon = iris.coords.DimCoord(
            self.lon(),
            standard_name='longitude', 
            units='degrees'
        )
        # Aux coords.
        # 850 hPa for temperature.
        p850 = iris.coords.AuxCoord(
            np.array([850]),
            standard_name='air_pressure',
            units='hPa'
        )
        # 500 hPa for geopotential.
        p500 = iris.coords.AuxCoord(
            np.array([500]),
            standard_name='air_pressure',
            units='hPa'
        )

        # Define cubes.
        # 2 metre temperature.
        t2m = iris.cube.Cube(t2m,
            long_name='2m_temperature',
            var_name='t2m',
            units='K',
            dim_coords_and_dims=[
                (time, 0), (lat, 1), (lon, 2)
            ],
            attributes={
                'source': 'AMSIMP Global Forecast Model',
            }
        ) 
        pbar.update()

        # 850 hPa temperature.
        t = iris.cube.Cube(t,
            standard_name='air_temperature',
            var_name='t',
            units='K',
            dim_coords_and_dims=[
                (time, 0), (lat, 1), (lon, 2)
            ],
            attributes={
                'source': 'AMSIMP Global Forecast Model',
            }
        )
        t.add_aux_coord(p850)
        pbar.update()

        # 500 hPa geopotential.
        z = iris.cube.Cube(z,
            standard_name='geopotential',
            var_name='z',
            units='m2 s-2',
            dim_coords_and_dims=[
                (time, 0), (lat, 1), (lon, 2)
            ],
            attributes={
                'source': 'AMSIMP Global Forecast Model',
            }
        )
        z.add_aux_coord(p500)
        pbar.update()

        # Finish progress bar.
        pbar.close()

        # Define output forecast.
        forecast = iris.cube.CubeList([t2m, t, z])

        return forecast
