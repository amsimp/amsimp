# Import packages.
import time
import sys
import os
import amsimp
import score
import xarray as xr
import iris
from iris.cube import CubeList
import numpy as np
from tqdm import tqdm
import warnings
import sys

# Suppress iris save warnings.
warnings.filterwarnings(action="ignore", category=UserWarning)

# Load data.
historical_data = iris.load("benchmark.nc")

# Extract parameters.
# Air temperature at 850 hPa.
t = historical_data.extract("t")[0]

# Geopotential at 500 hPa.
z = historical_data.extract("z")[0]

# Function to split data into windows (preprocessing).
def preprocess_data(dataset, past_history, future_target):
    X, y = list(), list()
    for i in range(dataset.shape[0]):
        # Find the end.
        end_ix = i + past_history
        out_end_ix = end_ix + future_target

        # Determine if we are beyond the dataset.
        if out_end_ix > dataset.shape[0]:
            break

        # Gather the input and output components.
        seq_x, seq_y = dataset[i:end_ix, :, :, :], dataset[end_ix:out_end_ix, :, :, :]

        # Append to list.
        X.append(seq_x)
        y.append(seq_y)

    return X, y


# Define progress bar (preprocessing).
pbar = tqdm(total=2, desc="Preprocessing historical data")

# Define the weather forecasting inputs and the observations after the fact.
# Air temperature.
input_t, obs_t = preprocess_data(t, 1, 60)
input_t, obs_t = input_t[::3], obs_t[::3]
pbar.update()

# Geopotential.
input_z, obs_z = preprocess_data(z, 1, 60)
input_z, obs_z = input_z[::3], obs_z[::3]
pbar.update()

# Progress bar finished (preprocessing).
pbar.close()

# Function to determine the skill and accuracy of the 5 day weather forecast produced by
# the software.
def accuracy(fct_cube, obs_cube):
    # Convert cube to xarray.
    # Observations.
    obs_xarray = xr.DataArray.from_iris(obs_cube)
    # Naive.
    naive_cube = fct_cube.copy()
    naive_cube.data = np.resize(naive_cube[0].data, naive_cube.shape)
    naive_xarray = xr.DataArray.from_iris(naive_cube)
    # Forecast.
    fct_cube = fct_cube[1:, :, :, :]
    fct_xarray = xr.DataArray.from_iris(fct_cube)

    # Forecast.
    # Anomaly Correlation Coefficient.
    acc = score.compute_weighted_acc(fct_xarray, obs_xarray)
    acc = acc.values
    # Root Mean Squared Error.
    rmse = score.compute_weighted_rmse(fct_xarray, obs_xarray)
    rmse = rmse.values
    # Mean Absolute Error.
    mae = score.compute_weighted_mae(fct_xarray, obs_xarray)
    mae = mae.values

    # Naive.
    # Anomaly Correlation Coefficient.
    naive_acc = score.compute_weighted_acc(naive_xarray, obs_xarray)
    naive_acc = naive_acc.values
    # Root Mean Squared Error.
    naive_rmse = score.compute_weighted_rmse(naive_xarray, obs_xarray)
    naive_rmse = naive_rmse.values
    # Mean Absolute Error.
    naive_mae = score.compute_weighted_mae(naive_xarray, obs_xarray)
    naive_mae = naive_mae.values

    return acc, rmse, mae, naive_acc, naive_rmse, naive_mae


# Determine the amount of time needed to generate a 5 day forecast.
performance = []

# Define loop length.
len_test = len(input_t)

# Store the skill and accuracy of the 5 day weather forecast produced by the
# software.
# Air temperature at 850 hPa.
accuracy_t = np.zeros((len_test, 6, 60))
# Geopotential at 500 hPa.
accuracy_z = np.zeros((len_test, 6, 60))

for i in range(len_test):
    # Create cube list of observations for AMSIMP.
    model_input = CubeList([input_t[i][0].copy(), input_z[i][0].copy()])

    # Save model input to file.
    iris.save(model_input, "temp.nc")

    # Define atmospheric state in AMSIMP.
    model = amsimp.OperationalModel(
        forecast_length=120,
        amsimp_ic=False,
        initialisation_conditions="temp.nc"
    )

    # Generating forecast.
    start = time.time()
    fct = model.generate_forecast()

    # Amount of time to generate forecast and append to performance list.
    runtime = time.time() - start
    performance.append(runtime)

    # Forecast parameters.
    # Air temperature at 850 hPa.
    fct_t = fct.extract("t")[0]

    # Geopotential at 500 hPa.
    fct_z = fct.extract("z")[0]

    # Interpolate observational cubes.
    lat, lon = np.load('lat.npy'), np.load('lon.npy')

    # Define grid points for interpolation.
    grid_points = [("latitude", lat), ("longitude", lon)]

    # Determine the skill and accuracy of the 5 day weather forecast produced by
    # the software.
    # Air temperature at 850 hPa.
    acc, rmse, mae, naive_acc, naive_rmse, naive_mae = accuracy(
        fct_t.interpolate(grid_points, iris.analysis.Linear()),
        obs_t[i]
        .copy()
        .interpolate(grid_points, iris.analysis.Linear()),
    )
    accuracy_t[i] = np.array([acc, rmse, mae, naive_acc, naive_rmse, naive_mae])

    # Geopotential at 500 hPa.
    acc, rmse, mae, naive_acc, naive_rmse, naive_mae = accuracy(
        fct_z.interpolate(grid_points, iris.analysis.Linear()),
        obs_z[i]
        .copy()
        .interpolate(grid_points, iris.analysis.Linear()),
    )
    accuracy_z[i] = np.array([acc, rmse, mae, naive_acc, naive_rmse, naive_mae])

    # Remove temporary file.
    os.remove("temp.nc")

    # Print progress.
    print("Progress: {}/{}".format(i + 1, len_test))

# Save results.
# Air temperature at 850 hPa.
np.save("results/t.npy", accuracy_t)
# Geopotential at 500 hPa.
np.save("results/z.npy", accuracy_z)
# Execution time.
np.save("results/performance.npy", np.asarray(performance))
