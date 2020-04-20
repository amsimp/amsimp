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

# -----------------------------------------------------------------------------------------#

# Creates a CSV file.
def csv_file():
    file = os.path.isfile("performance/performance.csv")
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

# Benchmark function.
def benchmarking():
    # CSV file.
    writer = csv_file()

    # Loop by each the number of forecast days.
    for num in range(3):
        # Determine whether to enable the recurrent neural network, and the
        # ensemble forecast system.
        if i == 0:
            detail = amsimp.Dynamics(efs=False, ai=False)
            label = "physical_model"
        elif i == 1:
            detail = amsimp.Dynamics(efs=False)
            label = "physical_model_with_rnn"
        elif i == 2:
            detail = amsimp.Dynamics()
            label = "physical_model_with_rnn_and_efs"

        # Current date.
        date = detail.date

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

        # Create directory.
        try:
            os.mkdir('accuracy/amsimp')
        except OSError:
            pass

        try:
            os.mkdir('accuracy/amsimp'+label)
        except OSError:
            pass

        try:
            os.mkdir('accuracy/amsimp'+label+'/'+year)
        except OSError:
            pass

        try:
            os.mkdir('accuracy/amsimp'+label+'/'+year+'/'+month)
        except OSError:
            pass

        try:
            os.mkdir('accuracy/amsimp'+label+'/'+year+"/"+month+'/'+day)
        except OSError:
            pass

        # Save forecast in this directory.
        folder = 'accuracy/amsimp'+label+'/'+year+"/"+month+'/'+day+'/
        filename = folder+'motus_aeris.nc'

        # Start timer.
        start = time.time()

        output = detail.atmospheric_prognostic_method()

        # Store runtime in variable.
        finish = time.time()
        runtime = finish - start

        # Write runtime into CSV file.
        write_data(writer, {"scheme": label, "time": runtime})

        # Save forecast.
        iris.save(output, filename)

    # Benchmark complete.
    print("Benchmarking Complete.")

# -----------------------------------------------------------------------------------------#

benchmarking()
