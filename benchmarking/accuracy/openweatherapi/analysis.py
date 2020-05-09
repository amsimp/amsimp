# Import dependices.
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Set starting date
date = datetime(2020, 4, 29, 0)

# Create one NumPy array for the initial conditions.
# Atmospheric Pressure.
initial_pressure = []

# Temperature.
# Air Temperature.
initial_T = []

# Virtual Temperature.
initial_Tv = []

# Relative Humidity
initial_humidity = []

# Wind.
# Zonal Wind.
initial_u = []

# Meridional Wind
initial_v = []

n_days = 6 * 4
for i in range(n_days):
    # Define the date.
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

    folder = year+"/"+month+"/"+day+"/"+hour+"/"

    # Atmospheric Pressure/
    pressure_file = folder + "pressure.npy"
    pressure = np.load(pressure_file)
    initial_pressure.append(pressure)

    # Temperature.
    # Air Temperature.
    T_file = folder + "temperature.npy"
    T = np.load(T_file)
    initial_T.append(T)

    # Virtual Temperature.
    Tv_file = folder + "virtual_temperature.npy"
    Tv = np.load(Tv_file)
    initial_Tv.append(Tv)

    # Relative Humidity
    humidity_file = folder + "humidity.npy"
    humidity = np.load(humidity_file)
    initial_humidity.append(humidity)

    # Wind.
    # Zonal Wind.
    u_file = folder + "zonal_wind.npy"
    u = np.load(u_file, allow_pickle=True)
    u = np.asarray(u, dtype=float)
    initial_u.append(u)

    # Meridional Wind.
    v_file = folder + "meridional_wind.npy"
    v = np.load(v_file, allow_pickle=True)
    v = np.asarray(v, dtype=float)
    initial_v.append(v)

    date = date + timedelta(hours=+6)

# Convert to NumPy arrays.
# Atmospheric Pressure.
initial_pressure = np.asarray(initial_pressure)
# Temperature.
# Air Temperture.
initial_T = np.asarray(initial_T)
# Virtual Temperature.
initial_Tv = np.asarray(initial_Tv)
# Relative Humidity
initial_humidity = np.asarray(initial_humidity)
# Wind.
# Zonal Wind.
initial_u = np.asarray(initial_u)
# Meridional Wind.
initial_v = np.asarray(initial_v)

# Accuracy Benchmarking.
# Residual Forecast Error
def forecast_error(prediction, actual):
    return actual - prediction

# Forecast Bias
def forecast_bias(prediction, actual):
    output = []
    len_prediction = len(prediction)
    for i in range(len_prediction):
        forecast_bias = np.mean(forecast_error(prediction[i], actual[i]))
        output.append(forecast_bias)
    return np.asarray(output)

# Mean Absolute Error
def mae(prediction, actual):
    output = []
    len_prediction = len(prediction)
    for i in range(len_prediction):
        mae = np.mean(np.abs(forecast_error(prediction[i], actual[i])))
        output.append(mae)
    return np.asarray(output)

# Mean Squared Error
def mse(prediction, actual):
    output = []
    len_prediction = len(prediction)
    for i in range(len_prediction):
        mse = np.mean(np.power(forecast_error(prediction[i], actual[i]), 2))
        output.append(mse)
    return np.asarray(output)

# Root Mean Squared Error
def rmse(prediction, actual):
    return np.sqrt(mse(prediction, actual))

# Mean Absolute Percentage Error.
def mape(prediction, actual):
    output = []
    len_prediction = len(prediction)
    for i in range(len_prediction):
        mape = np.mean(
            np.abs(
                forecast_error(prediction[i], actual[i])
            ) / np.abs(
                actual[i]
            )
        )
        output.append(mape)
    return np.asarray(output)

# Mean Absolute Scaled Error.
def mase(prediction, actual):
    mae_predict = mae(prediction, actual)
    naive_predict = np.zeros(prediction.shape) + actual[0]
    mae_naive = mae(naive_predict, actual)
    output = mae_predict / mae_naive
    return output

# Combine accuracy benchmarks into a single function.
def accuracy_benchmark(prediction, actual):
    output = []

    forecast_bias_value = forecast_bias(prediction, actual)
    mae_value = mae(prediction, actual)
    mse_value = mse(prediction, actual)
    rmse_value = rmse(prediction, actual)
    mape_value = mape(prediction, actual)
    mase_value = mase(prediction, actual)

    output.append(forecast_bias_value)
    output.append(mae_value)
    output.append(mse_value)
    output.append(rmse_value)
    output.append(mape_value)
    output.append(mase_value)

    return np.asarray(output)

# Redefine starting date
date = datetime(2020, 4, 29, 0)
# Labels for CSV.
# Columns for DataFrame.
measures_of_error = np.array(
    ['forecast_bias', 'mae', 'mse', 'rmse', 'mape', 'mase']
)
indices = np.linspace(0, 114, 20)

# Define the date.
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

folder = year+"/"+month+"/"+day+"/"+hour+"/"
error_folder = "error/"

# Atmospheric Pressure.
# Forecasted Pressure.
forecast_pressure_file = folder + 'forecast_pressure.npy'
forecast_pressure = np.load(forecast_pressure_file)
forecast_pressure = np.transpose(forecast_pressure, (2, 0, 1))
forecast_pressure = forecast_pressure[::2, :, :]
len_forecast = len(forecast_pressure)
# Actual Pressure.
actual_pressure = initial_pressure[:len_forecast, :, :]
# Test accuracy.
pressure_error = accuracy_benchmark(forecast_pressure, actual_pressure)
pressure_error = np.transpose(pressure_error)
pressure_df = pd.DataFrame(
    data=pressure_error, index=indices, columns=measures_of_error
)
pressure_df.index.name = "forecast_period"
# Save to CSV.
pressure_df.to_csv(error_folder+'pressure.csv')

# Temperature
# Air Temperature.
# Forecasted Temperature.
forecast_T_file = folder + 'forecast_temperature.npy'
forecast_T = np.load(forecast_T_file)
forecast_T = np.transpose(forecast_T, (2, 0, 1))
forecast_T = forecast_T[::2, :, :]
# Actual Temperature.
actual_T = initial_T[:len_forecast, :, :]
# Test accuracy.
T_error = accuracy_benchmark(forecast_T, actual_T)
T_error = np.transpose(T_error)
T_df = pd.DataFrame(
    data=T_error, index=indices, columns=measures_of_error
)
T_df.index.name = "forecast_period"
# Save to CSV.
T_df.to_csv(error_folder+'temperature.csv')
# Virtual Temperature.
# Forecasted Virtual Temperature.
forecast_Tv_file = folder + 'forecast_virtualtemperature.npy'
forecast_Tv = np.load(forecast_Tv_file)
forecast_Tv = np.transpose(forecast_Tv, (2, 0, 1))
forecast_Tv = forecast_Tv[::2, :, :]
# Actual Virtual Temperature.
actual_Tv = initial_Tv[:len_forecast, :, :]
# Test accuracy.
Tv_error = accuracy_benchmark(forecast_Tv, actual_Tv)
Tv_error = np.transpose(Tv_error)
Tv_df = pd.DataFrame(
    data=Tv_error, index=indices, columns=measures_of_error
)
Tv_df.index.name = "forecast_period"
# Save to CSV.
Tv_df.to_csv(error_folder+'virtual_temperature.csv')

# Relative Humidity.
# Forecasted Relative Humidity.
forecast_humidity_file = folder + 'forecast_humidity.npy'
forecast_humidity = np.load(forecast_humidity_file)
forecast_humidity = np.transpose(forecast_humidity, (2, 0, 1))
forecast_humidity = forecast_humidity[::2, :, :]
# Actual Pressure.
actual_humidity = initial_humidity[:len_forecast, :, :]
# Test accuracy.
humidity_error = accuracy_benchmark(forecast_humidity, actual_humidity)
humidity_error = np.transpose(humidity_error)
humidity_df = pd.DataFrame(
    data=humidity_error, index=indices, columns=measures_of_error
)
humidity_df.index.name = "forecast_period"
# Save to CSV.
humidity_df.to_csv(error_folder+'humidity.csv')

# Wind
# Zonal Wind.
# Forecasted Zonal Wind.
forecast_u_file = folder + 'forecast_zonalwind.npy'
forecast_u = np.load(forecast_u_file)
forecast_u = np.asarray(forecast_u, dtype=float)
forecast_u = np.transpose(forecast_u, (2, 0, 1))
forecast_u = forecast_u[::2, :, :]
# Actual Zonal Wind.
actual_u = initial_u[:len_forecast, :, :]
# Test accuracy.
u_error = accuracy_benchmark(forecast_u, actual_u)
u_error = np.transpose(u_error)
u_df = pd.DataFrame(
    data=u_error, index=indices, columns=measures_of_error
)
u_df.index.name = "forecast_period"
# Save to CSV.
u_df.to_csv(error_folder+'zonal_wind.csv')

# Meridional Wind.
# Forecasted Meridional Wind.
forecast_v_file = folder + 'forecast_meridionalwind.npy'
forecast_v = np.load(forecast_v_file)
forecast_v = np.asarray(forecast_v, dtype=float)
forecast_v = np.transpose(forecast_v, (2, 0, 1))
forecast_v = forecast_v[::2, :, :]
# Actual Meridional Wind.
actual_v = initial_v[:len_forecast, :, :]
# Test accuracy.
v_error = accuracy_benchmark(forecast_v, actual_v)
v_error = np.transpose(v_error)
v_df = pd.DataFrame(
    data=v_error, index=indices, columns=measures_of_error
)
v_df.index.name = "forecast_period"
# Save to CSV.
v_df.to_csv(error_folder+'meridional_wind.csv')

date = date + timedelta(hours=+6)

# Plotting.
# Function.
def plot(x, y, title, metric):
    plt.plot(x, y)
    plt.scatter(x, y, color='green')
    plt.xlabel("Forecast Period (Hours)")
    plt.ylabel(metric)
    plt.title("OpenWeatherAPI "+title+" "+metric)
    folder = title.lower()
    folder = folder.replace(" ", "_")
    folder = "plots/"+folder+"/"
    filename = metric.lower()
    filename = filename.replace(" ", "_")
    plt.savefig(folder+filename, dpi=300)
    plt.close()

def label_decide(num):
    if i == 0:
        label = 'Forecast Bias'
    elif i == 1:
        label = 'Mean Absolute Error'
    elif i == 2:
        label = 'Mean Squared Error'
    elif i == 3:
        label = 'Root Mean Squared Error'
    elif i == 4:
        label = 'Mean Absolute Percentage Error'
    elif i == 5:
        label = 'Mean Absolute Scaled Error'
    return label

# Atmospheric Pressure.
pressure_error = np.transpose(pressure_error)
for i in range(len(pressure_error)):
    metric = label_decide(i)
    title = "Atmospheric Pressure"
    if i != 4: 
        plot(indices[1:], pressure_error[i, 1:], title, metric)
    else:
        plot(indices[1:], (pressure_error[i, 1:] * 100), title, metric)

# Temperature.
# Air Temperature.
T_error = np.transpose(T_error)
for i in range(len(T_error)):
    metric = label_decide(i)
    title = "Air Temperature"
    if i != 4: 
        plot(indices[1:], T_error[i, 1:], title, metric)
    else:
        plot(indices[1:], (T_error[i, 1:] * 100), title, metric)
# Virtual Temperature.
Tv_error = np.transpose(Tv_error)
for i in range(len(T_error)):
    metric = label_decide(i)
    title = "Virtual Temperature"
    if i != 4: 
        plot(indices[1:], Tv_error[i, 1:], title, metric)
    else:
        plot(indices[1:], (Tv_error[i, 1:] * 100), title, metric)

# Relative Humidity.
humidity_error = np.transpose(humidity_error)
for i in range(len(humidity_error)):
    metric = label_decide(i)
    title = "Relative Humidity"
    if i != 4: 
        plot(indices[1:], humidity_error[i, 1:], title, metric)
    else:
        plot(indices[1:], (humidity_error[i, 1:] * 100), title, metric)

# Wind.
# Zonal Wind.
u_error = np.transpose(u_error)
for i in range(len(u_error)):
    metric = label_decide(i)
    title = "Zonal Wind"
    if i != 4: 
        plot(indices[1:], humidity_error[i, 1:], title, metric)
    else:
        plot(indices[1:], (humidity_error[i, 1:] * 100), title, metric)
# Meridional Wind.
v_error = np.transpose(v_error)
for i in range(len(v_error)):
    metric = label_decide(i)
    title = "Meridional Wind"
    if i != 4: 
        plot(indices[1:], v_error[i, 1:], title, metric)
    else:
        plot(indices[1:], (v_error[i, 1:] * 100), title, metric)
