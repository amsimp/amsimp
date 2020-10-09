# Import packages.
import sys
import xskillscore as xs
import xarray as xr
import iris
from iris.cube import CubeList
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from astropy import units

# Load data.
historical_data = iris.load('historical-data/*.nc')

# Load the various parameters as NumPy arrays.
# Air temperature.
temperature = historical_data.extract("air_temperature")[0][::3]
# Relative humidity.
relative_humidity = historical_data.extract("relative_humidity")[0][::3]
# Geopotential.
geopotential = historical_data.extract("geopotential")[0][::3]
# Wind.
# Zonal.
zonal_wind = historical_data.extract("eastward_wind")[0][::3]
zonal_wind.standard_name = 'x_wind'
# Meridional.
meridional_wind = historical_data.extract("northward_wind")[0][::3]
meridional_wind.standard_name = 'y_wind'

# Unavailable GFS dates.
unavailable_gfs_dates = [
    "20190202",
    "20190204",
    "20190419",
    "20190427",
    "20190519",
    "20190614",
    "20190616",
    "20190618",
    "20190904",
    "20191109"
]

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
obs_temperature = preprocess_data(temperature, 1, (5*4))[1]
obs_temperature = obs_temperature[::8]
t.update(1)
# Relative humidity.
obs_relativehumidity = preprocess_data(relative_humidity, 1, (5*4))[1]
obs_relativehumidity = obs_relativehumidity[::8]
t.update(1)
# Geopotential.
obs_geopotential = preprocess_data(geopotential, 1, (5*4))[1]
obs_geopotential = obs_geopotential[::8]
t.update(1)
# Wind.
# Zonal.
obs_zonalwind = preprocess_data(zonal_wind, 1, (5*4))[1]
obs_zonalwind = obs_zonalwind[::8]
t.update(1)
# Meridional.
obs_meridionalwind = preprocess_data(meridional_wind, 1, (5*4))[1]
obs_meridionalwind = obs_meridionalwind[::8]
t.update(1)

# Progress bar finished (preprocessing).
t.close()

# Function to determine the skill and accuracy of the 5 day weather forecast produced by
# the software.
def accuracy(fct_cube, obs_cube):
    naive_data = fct_cube.data[0]
    
    # Convert cube to xarray.
    # Forecast.
    fct_xarray = xr.DataArray.from_iris(fct_cube)
    # Observations.
    obs_cube.coords('pressure_level')[0].var_name = 'pressure_level'
    obs_cube = obs_cube[:, :, ::-1, :]
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
    obs_max = np.mean(np.mean(obs_max, axis=3), axis=2)
    obs_min = np.resize(np.min(obs_data, axis=0), obs_data.shape)
    obs_min = np.mean(np.mean(obs_min, axis=3), axis=2)
    nrmse = xs.rmse(
        obs_xarray, fct_xarray, dim=["latitude", "longitude"]
    ).values / (obs_max - obs_min)
    nrmse = np.mean(nrmse, axis=1)
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

# Store the skill and accuracy of the 5 day weather forecast produced by the
# software.
# Air temperature.
accuracy_temperature = []
# Relative humidity.
accuracy_relativehumidity = []
# Geopotential.
accuracy_geopotential = []
# Wind.
# Zonal.
accuracy_zonalwind = []
# Meridional.
accuracy_meridionalwind = []

# Start date.
date = datetime(2019, 1, 1, 0)

# End date.
end = obs_temperature[-1][0].coord('time')
end = [cell.point for cell in end.cells()]
end = end[0]

# Define gravitational constant.
g = 9.80665
geo_unit = units.m**2 / units.s**2

i = 0
while date < end:
    # Check if forecast for date is available.
    str_date = date.strftime('%Y%m%d')
    check_availability = str_date in unavailable_gfs_dates

    if not check_availability:
        # Define observations after the fact.
        # Air temperature.
        obs_temp = obs_temperature[i]
        # Relative humidity.
        obs_rh = obs_relativehumidity[i]
        # Geopotential.
        obs_geo = obs_geopotential[i]
        # Wind.
        # Zonal.
        obs_u = obs_zonalwind[i]
        # Meridional.
        obs_v = obs_meridionalwind[i]

        # Load forecast.
        fct = iris.load('gfs-forecasts/' + str_date + ".nc")

        # Load the various parameters.
        # Air temperature.
        fct_temp = fct.extract("air_temperature")[0]
        # Relative humidity.
        fct_rh = fct.extract("relative_humidity")[0]
        # Geopotential height. 
        fct_height = fct.extract("geopotential_height")[0]
        # Geopotential.
        fct_geo = fct_height * g
        fct_geo.standard_name = "geopotential"
        fct_geo.units = geo_unit
        # Wind.
        # Zonal.
        fct_u = fct.extract("x_wind")[0]
        # Meridional.
        fct_v = fct.extract("y_wind")[0]

        # Determine the skill and accuracy of the 5 day weather forecast produced by
        # the software.
        # Air temperature.
        temp_r, temp_rmse, temp_nrmse, temp_mse, temp_mae, temp_mase = accuracy(fct_temp, obs_temp) 
        accuracy_temp = np.array([temp_r, temp_rmse, temp_nrmse, temp_mse, temp_mae, temp_mase])
        accuracy_temperature.append(accuracy_temp)
        # Relative humidity.
        rh_r, rh_rmse, rh_nrmse, rh_mse, rh_mae, rh_mase = accuracy(fct_rh, obs_rh)
        accuracy_rh = np.array([rh_r, rh_rmse, rh_nrmse, rh_mse, rh_mae, rh_mase])
        accuracy_relativehumidity.append(accuracy_rh)
        # Geopotential.
        r_geo, rmse_geo, nrmse_geo, mse_geo, mae_geo, mase_geo = accuracy(fct_geo, obs_geo)
        accuracy_geo = np.array([r_geo, rmse_geo, nrmse_geo, mse_geo, mae_geo, mase_geo])
        accuracy_geopotential.append(accuracy_geo)
        # Wind.
        # Zonal.
        r_u, rmse_u, nrmse_u, mse_u, mae_u, mase_u = accuracy(fct_u, obs_u)
        accuracy_u = np.array([r_u, rmse_u, nrmse_u, mse_u, mae_u, mase_u])
        accuracy_zonalwind.append(accuracy_u)
        # Meridional.
        r_v, rmse_v, nrmse_v, mse_v, mae_v, mase_v = accuracy(fct_v, obs_v)
        accuracy_v = np.array([r_v, rmse_v, nrmse_v, mse_v, mae_v, mase_v])
        accuracy_meridionalwind.append(accuracy_v)
        
        print("Progress: " + str_date)
    
    # Add time.
    date = date + timedelta(days=+2)
    i += 1

# Save results.
# Accuracy.
# Air temperature.
np.save('gfs_results/temperature.npy', np.asarray(accuracy_temperature))
# Geopotential.
np.save('gfs_results/geopotential.npy', np.asarray(accuracy_geopotential))
# Relative humidity.
np.save('gfs_results/relative_humidity.npy', np.asarray(accuracy_relativehumidity))
# Wind.
# Zonal.
np.save('gfs_results/zonal_wind.npy', np.asarray(accuracy_zonalwind))
# Meridional.
np.save('gfs_results/meridional_wind.npy', np.asarray(accuracy_meridionalwind))
