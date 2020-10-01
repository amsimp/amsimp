# Import packages.
import time
import sys
import amsimp
import xskillscore as xs
import xarray as xr
import iris
from iris.cube import CubeList
import numpy as np
from tqdm import tqdm

# Load data.
historical_data = iris.load('historical-data/*.nc')

# Load the various parameters as NumPy arrays.
# Air temperature.
temperature = historical_data.extract("air_temperature")[0]
# Relative humidity.
relative_humidity = historical_data.extract("relative_humidity")[0]
# Geopotential.
geopotential = historical_data.extract("geopotential")[0]
# Wind.
# Zonal.
zonal_wind = historical_data.extract("eastward_wind")[0]
zonal_wind.standard_name = 'x_wind'
# Meridional.
meridional_wind = historical_data.extract("northward_wind")[0]
meridional_wind.standard_name = 'y_wind'

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
t = tqdm(total=5, desc='Preprocessing historical data')

# Define the weather forecasting inputs and the observations after the fact.
# Air temperature.
input_temperature, obs_temperature = preprocess_data(temperature, (15*12), (5*12))
input_temperature, obs_temperature = input_temperature[::24], obs_temperature[::24]
t.update(1)
# Relative humidity.
input_relativehumidity, obs_relativehumidity = preprocess_data(relative_humidity, (15*12), (5*12))
input_relativehumidity, obs_relativehumidity = input_relativehumidity[::24], obs_relativehumidity[::24]
t.update(1)
# Geopotential.
input_geopotential, obs_geopotential = preprocess_data(geopotential, (15*12), (5*12))
input_geopotential, obs_geopotential = input_geopotential[::24], obs_geopotential[::24]
t.update(1)
# Wind.
# Zonal.
input_zonalwind, obs_zonalwind = preprocess_data(zonal_wind, (15*12), (5*12))
input_zonalwind, obs_zonalwind = input_zonalwind[::24], obs_zonalwind[::24]
t.update(1)
# Meridional.
input_meridionalwind, obs_meridionalwind = preprocess_data(meridional_wind, (15*12), (5*12))
input_meridionalwind, obs_meridionalwind = input_meridionalwind[::24], obs_meridionalwind[::24]
t.update(1)

# Progress bar finished (preprocessing).
t.close()

# Function to determine the skill and accuracy of the 5 day weather forecast produced by
# the software.
def accuracy(fct_cube, obs_cube):
    naive_data = fct_cube.data[0]
    
    # Convert cube to xarray.
    # Forecast.
    fct_cube = fct_cube[1:, :, :, :]
    fct_xarray = xr.DataArray.from_iris(fct_cube)
    # Observations.
    obs_cube.coords('pressure_level')[0].var_name = 'pressure_level'
    obs_cube = obs_cube[:, ::-1, ::-1, :]
    obs_data = obs_cube.data
    obs_newcube = fct_cube.copy()
    obs_newcube.data = obs_data
    obs_newcube.metadata.attributes['source'] = 'ECMWF'
    obs_cube = obs_newcube
    obs_xarray = xr.DataArray.from_iris(obs_cube)
    # Naïve forecast.
    naive_data = np.resize(naive_data, obs_data.shape)
    naive_cube = fct_cube.copy()
    naive_cube.data = naive_data
    naive_cube.metadata.attributes['source'] = None
    naive_xarray = xr.DataArray.from_iris(naive_cube)

    # Pearson Correlation
    r = xs.pearson_r(
        obs_xarray, fct_xarray, dim=["pressure_level", "latitude", "longitude"]
    )
    r = r.values
    # Root Mean Squared Error.
    rmse = xs.rmse(
        obs_xarray, fct_xarray, dim=["pressure_level", "latitude", "longitude"]
    )
    rmse = rmse.values
    # Normalised Root Mean Squared Error.
    obs_max = np.resize(np.max(obs_data, axis=0), obs_data.shape)
    obs_min = np.resize(np.min(obs_data, axis=0), obs_data.shape)
    nrmse = rmse / (obs_max - obs_min)
    # Mean Squared Error.
    mse = xs.mse(
        obs_xarray, fct_xarray, dim=["pressure_level", "latitude", "longitude"]
    )
    mse = mse.values
    # Mean Absolute Error.
    mae = xs.mae(
        obs_xarray, fct_xarray, dim=["pressure_level", "latitude", "longitude"]
    )
    mae = mae.values
    # Mean Absolute Scaled Error.
    mae_naive = xs.mae(
        obs_xarray, naive_xarray, dim=["pressure_level", "latitude", "longitude"]
    )
    mae_naive = mae_naive.values
    mase = mae / mae_naive

    return r, rmse, nrmse, mse, mae, mase

# Determine the amount of time needed to generate a 5 day forecast.
performance = []

# Define loop length.
len_test = len(input_temperature)

# Store the skill and accuracy of the 5 day weather forecast produced by the
# software.
# Air temperature.
accuracy_temperature = np.zeros((len_test, 6, (5*12)))
# Relative humidity.
accuracy_relativehumidity = np.zeros((len_test, 6, (5*12)))
# Geopotential.
accuracy_geopotential = np.zeros((len_test, 6, (5*12)))
# Wind.
# Zonal.
accuracy_zonalwind = np.zeros((len_test, 6, (5*12)))
# Meridional.
accuracy_meridionalwind = np.zeros((len_test, 6, (5*12)))

for i in range(len_test):
    # Define progress bar (generating inputs).
    t = tqdm(total=5, desc='Generating inputs')

    # Define current input, and the observations after the fact.
    # Air temperature.
    input_temp, obs_temp = input_temperature[i], obs_temperature[i]
    t.update(1)
    # Relative humidity.
    input_rh, obs_rh = input_relativehumidity[i], obs_relativehumidity[i]
    t.update(1)
    # Geopotential.
    input_geo, obs_geo = input_geopotential[i], obs_geopotential[i]
    t.update(1)
    # Wind.
    # Zonal.
    input_u, obs_u = input_zonalwind[i], obs_zonalwind[i]
    t.update(1)
    # Meridional.
    input_v, obs_v = input_meridionalwind[i], obs_meridionalwind[i]
    t.update(1)
    
    # Progress bar finished (generating inputs).
    t.close()   

    # Create cube list of observations for AMSIMP.
    input_data = CubeList([input_temp, input_rh, input_geo, input_u, input_v])

    # Define atmospheric state in AMSIMP.
    state = amsimp.Weather(historical_data=input_data)

    # Generating forecast.
    start = time.time()
    fct = state.generate_forecast()

    # Amount of time to generate forecast and append to performance list.
    runtime = time.time() - start
    performance.append(runtime)

    # Define the weather forecast for the the various parameters.
    # Define progress bar (postprocessing).
    t = tqdm(total=5, desc='Post-processing historical data')

    # Load the various parameters as NumPy arrays.
    # Air temperature.
    fct_temp = fct.extract("air_temperature")[0]
    t.update(1)
    # Relative humidity.
    fct_rh = fct.extract("relative_humidity")[0]
    t.update(1)
    # Geopotential.
    fct_geo = fct.extract("geopotential")[0]
    t.update(1)
    # Wind.
    # Zonal.
    fct_u = fct.extract("x_wind")[0]
    t.update(1)
    # Meridional.
    fct_v = fct.extract("y_wind")[0]
    t.update(1)

    # Progress bar finished (postprocessing).
    t.close()

    # Determine the skill and accuracy of the 5 day weather forecast produced by
    # the software.
    # Air temperature.
    temp_r, temp_rmse, temp_nrmse, temp_mse, temp_mae, temp_mase = accuracy(fct_temp, obs_temp) 
    accuracy_temp = np.array([temp_r, temp_rmse, temp_nrmse, temp_mse, temp_mae, temp_mase])
    accuracy_temperature[i] = accuracy_temp
    # Relative humidity.
    rh_r, rh_rmse, rh_nrmse, rh_mse, rh_mae, rh_mase = accuracy(fct_rh, obs_rh)
    accuracy_rh = np.array([rh_r, rh_rmse, rh_nrmse, rh_mse, rh_mae, rh_mase])
    accuracy_relativehumidity[i] = accuracy_rh
    # Geopotential.
    r_geo, rmse_geo, nrmse_geo, mse_geo, mae_geo, mase_geo = accuracy(fct_geo, obs_geo)
    accuracy_geo = np.array([r_geo, rmse_geo, nrmse_geo, mse_geo, mae_geo, mase_geo])
    accuracy_geopotential[i] = accuracy_geo
    # Wind.
    # Zonal.
    r_u, rmse_u, nrmse_u, mse_u, mae_u, mase_u = accuracy(fct_u, obs_u)
    accuracy_u = np.array([r_u, rmse_u, nrmse_u, mse_u, mae_u, mase_u])
    accuracy_zonalwind[i] = accuracy_u
    # Meridional.
    r_v, rmse_v, nrmse_v, mse_v, mae_v, mase_v = accuracy(fct_v, obs_v)
    accuracy_v = np.array([r_v, rmse_v, nrmse_v, mse_v, mae_v, mase_v])
    accuracy_meridionalwind[i] = accuracy_v
    
    print("Progress: " + str(i+1) + "/" + str(len_test))

# Save results.
# Performance.
np.save('results/performance.npy', np.asarray(performance))
# Accuracy.
# Air temperature.
np.save('results/temperature.npy', accuracy_temperature)
# Geopotential.
np.save('results/geopotential.npy', accuracy_geopotential)
# Relative humidity.
np.save('results/relative_humidity.npy', accuracy_relativehumidity)
# Wind.
# Zonal.
np.save('results/zonal_wind.npy', accuracy_zonalwind)
# Meridional.
np.save('results/meridional_wind.npy', accuracy_meridionalwind)