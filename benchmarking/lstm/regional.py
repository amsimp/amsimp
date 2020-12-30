# Import packages.
import time
import sys
import amsimp
import score
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
input_temperature, obs_temperature = input_temperature[::(10*12)], obs_temperature[::(10*12)]
t.update(1)
# Relative humidity.
input_relativehumidity, obs_relativehumidity = preprocess_data(relative_humidity, (15*12), (5*12))
input_relativehumidity, obs_relativehumidity = input_relativehumidity[::(10*12)], obs_relativehumidity[::(10*12)]
t.update(1)
# Geopotential.
input_geopotential, obs_geopotential = preprocess_data(geopotential, (15*12), (5*12))
input_geopotential, obs_geopotential = input_geopotential[::(10*12)], obs_geopotential[::(10*12)]
t.update(1)
# Wind.
# Zonal.
input_zonalwind, obs_zonalwind = preprocess_data(zonal_wind, (15*12), (5*12))
input_zonalwind, obs_zonalwind = input_zonalwind[::(10*12)], obs_zonalwind[::(10*12)]
t.update(1)
# Meridional.
input_meridionalwind, obs_meridionalwind = preprocess_data(meridional_wind, (15*12), (5*12))
input_meridionalwind, obs_meridionalwind = input_meridionalwind[::(10*12)], obs_meridionalwind[::(10*12)]
t.update(1)

# Progress bar finished (preprocessing).
t.close()

# Function to determine the skill and accuracy of the 5 day weather forecast produced by
# the software.
def accuracy(fct_cube, obs_cube):    
    # Convert cube to xarray.
    # Observations.
    obs_cube.coords('pressure_level')[0].var_name = 'pressure_level'
    obs_cube = obs_cube[:, ::-1, ::-1, :]
    obs_xarray = xr.DataArray.from_iris(obs_cube)
    # Forecast.
    fct_cube = fct_cube[1:, :, :, :]
    fct_cube = fct_cube.regrid(obs_cube, iris.analysis.Linear())
    fct_xarray = xr.DataArray.from_iris(fct_cube)

    # Root Mean Squared Error.
    rmse = score.compute_weighted_rmse(fct_xarray, obs_xarray)
    rmse = rmse.values
    # Mean absolute error.
    mae = score.compute_weighted_mae(fct_xarray, obs_xarray)
    mae = mae.values

    return rmse, mae

# Determine the amount of time needed to generate a 5 day forecast.
performance = []

# Define loop length.
len_test = len(input_temperature)

# Store the skill and accuracy of the 5 day weather forecast produced by the
# software.
# Air temperature at 1000 hPa.
accuracy_temperature = np.zeros((len_test, 2, (5*12)))
level_1000 = iris.Constraint(pressure_level=1000)

for i in range(len_test):
    # Define current input, and the observations after the fact.
    # Air temperature.
    input_temp, obs_temp = input_temperature[i], obs_temperature[i]
    # Relative humidity.
    input_rh, obs_rh = input_relativehumidity[i], obs_relativehumidity[i]
    # Geopotential.
    input_geo, obs_geo = input_geopotential[i], obs_geopotential[i]
    # Wind.
    # Zonal.
    input_u, obs_u = input_zonalwind[i], obs_zonalwind[i]
    # Meridional.
    input_v, obs_v = input_meridionalwind[i], obs_meridionalwind[i]

    # Extract air temperature at 1000 hPa.
    # Air temperature.
    obs_temp = obs_temp.extract(level_1000)

    # Create cube list of observations for AMSIMP.
    model_input = CubeList([input_temp, input_rh, input_geo, input_u, input_v])

    # Define atmospheric state in AMSIMP.
    state = amsimp.Weather(historical_data=model_input, forecast_length=168)

    # Generating forecast.
    fct = state.generate_forecast()

    # Load the various parameters.
    # Air temperature at 1000 hPa.
    fct_temp = fct.extract("air_temperature")[0]
    fct_temp = fct_temp.extract(level_1000)

    # Determine the skill and accuracy of the 5 day weather forecast produced by
    # the software.
    # Air temperature.
    rmse, mae = accuracy(fct_temp, obs_temp)
    accuracy_temp = np.array([rmse, mae])
    accuracy_temperature[i] = accuracy_temp
    
    print("Progress: " + str(i+1) + "/" + str(len_test))

# Save results.
# Accuracy.
# Air temperature.
np.save('results/t1000.npy', accuracy_temperature)
