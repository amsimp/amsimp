# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import amsimp
import iris
import time
from datetime import datetime
from datetime import timedelta
import csv
import os
import random
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------------------#

# Creates a CSV file.
def csv_file():
    file = os.path.isfile("benchmarking/performance/performance.csv")
    csvfile = open(filename, "a")

    fieldnames = ["forecast_days", "time"]
    writer = csv.DictWriter(
        csvfile, delimiter=",", lineterminator="\n", fieldnames=fieldnames
    )

    if not file:
        writer.writeheader()

    return writer

# Write data to CSV file.
def write_data(writer, data):
    writer.writerow(data)

# -----------------------------------------------------------------------------------------#

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

# Combine accuracy benchmarks into a single function.
def accuracy_benchmark(prediction, actual):
    output = []

    forecast_bias = forecast_bias(prediction, actual)
    mae = mae(prediction, actual)
    mse = mse(prediction, actual)
    rmse = rmse(prediction, actual)

    output.append(forecast_bias)
    output.append(mae)
    output.append(mse)
    output.append(rmse)

    return np.asarray(output)

# -----------------------------------------------------------------------------------------#

# Benchmark function.
def benchmarking():
    # CSV file.
    writer = csv_file()
    # Starting date.
    start_date = datetime(2020, 4, 15, 0)
    date = start_date + timedelta(days=-30)
    # End of forecast period.
    max_date = start_date + timedelta(days=+5)
    # Historical Dataset.
    detail = amsimp.RNN(input_date=max_date, data_size=40)
    historical_data = detail.download_historical_data()
    # Reshape Historical Dataset.
    # Dimensions.
    len_time = historical_data[0].shape[0]
    len_pressure = len(detail.pressure_surfaces())
    len_lat = len(detail.latitude_lines())
    len_lon = len(detail.longitude_lines())

    def reshape(dataset):
        dataset = dataset.reshape(
            len_time, len_pressure, len_lat, len_lon
        )
        return dataset

    # Historical Dataset.
    # Geopotential Height.
    geopotential_height = reshape(historical_data[1])
    # Temperature.
    # Air Temperature.
    temperature = reshape(historical_data[0])
    # Relative Humidity.
    relative_humidity = reshape(historical_data[1])
    
    # Loop through days.
    n = 0
    zonal_wind = []
    meridional_wind = []
    virtual_temperature = []
    while n < len(geopotential_height):
        # Configure Wind Class.
        config = amsimp.Wind(
            input_data=True,
            geo=geopotential_height[n],
            temp=temperature[n],
            rh=relative_humidity[n]
        )
        # Geostrophic Wind.
        # Zonal Wind.
        zonalwind = config.zonal_wind()
        # Meridional Wind.
        meridionalwind = config.meridional_wind()
        # Virtual Temperature.
        virtualtemperature = config.virtual_temperature()

        # Append to List.
        zonal_wind.append(zonalwind)
        meridional_wind.append(meridionalwind)
        virtual_temperature.append(virtualtemperature)

        n += 1

    # Convert to NumPy arrays.
    zonal_wind = np.asarray(zonal_wind)
    meridional_wind = np.asarray(meridional_wind)
    virtual_temperature = np.asarray(virtual_temperature)

    n = 30 * 4
    # Benchmark on the last 30 days of data.
    for i in range(n):
        for num in range(3):
            # Determine whether to enable the recurrent neural network, and the
            # ensemble forecast system.
            if num == 0:
                detail = amsimp.Dynamics(input_date=date, forecast_length=120, efs=False, ai=False)
                label = "physical_model"
            elif num == 1:
                detail = amsimp.Dynamics(input_date=date, forecast_length=120, efs=False)
                label = "physical_model_with_rnn"
            elif num == 2:
                detail = amsimp.Dynamics(input_date=date, forecast_length=120)
                label = "physical_model_with_rnn_and_efs"

            # Start timer.
            start = time.time()

            output = detail.atmospheric_prognostic_method()

            # Indices for DataFrame.
            indices = detail.forecast_period().value[::6]
            # Columns for DataFrame.
            measures_of_error = np.array(['forecast_bias', 'mae', 'mse', 'rmse'])

            detail.remove_all_files()

            # Forecasted data.
            # Geopotential Height.
            height = output[0].data
            height = height[:, ::6, :, :, :]
            len_forecast = height.shape[1] + i
            # Geostrophic Wind.
            # Zonal Wind.
            u = output[1].data
            u = u[:, ::6, :, :, :]
            # Meridional Wind.
            v = output[2].data
            v = v[:, ::6, :, :, :]
            # Temperature.
            # Air Temperature.
            temp = output[5].data
            temp = temp[:, ::6, :, :, :]
            # Virtual Temperature.
            temp_v = output[6].data
            temp_v = temp_v[:, ::6, :, :, :]
            # Relative Humidity.
            rh = output[7].data
            rh = rh[:, ::6, :, :, :]

            # Store runtime in variable.
            finish = time.time()
            runtime = finish - start

            # Write runtime into CSV file.
            write_data(writer, {"scheme": label, "time": runtime})

            # Accuracy Benchmarking.
            # Filter historical data.
            # Geopotential Height.
            actual_height = geopotential_height[i:len_forecast, :, :, :]
            # Geostrophic Wind.
            # Zonal Wind.
            actual_u = zonal_wind[i:len_forecast, :, :, :]
            # Meridional Wind.
            actual_v = meridional_wind[i:len_forecast, :, :, :]
            # Temperature.
            # Air Temperature.
            actual_temp = temperature[i:len_forecast, :, :, :]
            # Virtual Temperature.
            actual_tempv = virtual_temperature[i:len_forecast, :, :, :]
            # Relative Humidity.
            actual_rh = relative_humidity[i:len_forecast, :, :, :]
            
            for something in range(len(height)):
                # Geopotential Height.
                height_error = accuracy_benchmark(height[something], actual_height)
                height_error = np.transpose(height_error)
                height_df = pd.DataFrame(
                    data=height_error, index=indices, columns=measures_of_error
                )
                # Geostrophic Wind.
                # Zonal Wind.
                u_error = accuracy_benchmark(u[something], actual_u)
                u_error = np.transpose(u_error)
                u_df = pd.DataFrame(
                    data=u_error, index=indices, columns=measures_of_error
                )
                # Meridional Wind.
                v_error = accuracy_benchmark(v[something], actual_v)
                v_error = np.transpose(v_error)
                v_df = pd.DataFrame(
                    data=v_error, index=indices, columns=measures_of_error
                )
                # Temperature.
                # Air Temperature.
                temp_error = accuracy_benchmark(temp[something], actual_temp)
                temp_error = np.transpose(temp_error)
                temp_df = pd.DataFrame(
                    data=temp_error, index=indices, columns=measures_of_error
                )
                # Virtual Temperature.
                tempv_error = accuracy_benchmark(temp_v[something], actual_tempv)
                tempv_error = np.transpose(tempv_error)
                tempv_df = pd.DataFrame(
                    data=tempv_error, index=indices, columns=measures_of_error
                )
                # Relative Humidity.
                rh_error = accuracy_benchmark(rh[something], actual_rh)
                rh_error = np.transpose(rh_error)
                rh_df = pd.DataFrame(
                    data=rh_error, index=indices, columns=measures_of_error
                )

                # Output folder and file name.
                if i != 2:
                    folder = "benchmarking/accuracy/amsimp/"+label+"/"
                else:
                    folder = "benchmarking/accuracy/amsimp/"+label+"/"+str(something)+"/"

                # Save Results.
                height_df.to_csv(folder+"geopotential_height/"+date.strftime("%Y-%m-%d")+".csv")
                u_df.to_csv(folder+"zonal_wind/"+date.strftime("%Y-%m-%d")+".csv")
                v_df.to_csv(folder+"meridional_wind/"+date.strftime("%Y-%m-%d")+".csv")
                temp_df.to_csv(folder+"temperature/"+date.strftime("%Y-%m-%d")+".csv")
                tempv_df.to_csv(folder+"virtual_temperature/"+date.strftime("%Y-%m-%d")+".csv")
                rh_df.to_csv(folder+"relative_humidity/"+date.strftime("%Y-%m-%d")+".csv") 
        
        print(date)
        # Add 6 hours onto time.
        date = date + timedelta(days=+60)

    # Benchmark complete.
    print("Benchmarking Complete.")

# -----------------------------------------------------------------------------------------#

benchmarking()
