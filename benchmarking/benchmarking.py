# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import amsimp
import time
from datetime import datetime
import dateutil.relativedelta as future
import csv
import os
import numpy as np
import random

# -----------------------------------------------------------------------------------------#

# Output filename.
filename = input("The name of the CSV file: ")
filename = filename + ".csv"

# Creates a CSV file.
def csv_file():
    file = os.path.isfile(filename)
    csvfile = open(filename, "a")

    fieldnames = ["detail_level", "time"]
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

# Number of samples for each level of detail.
samples = int(input("The number of samples for each level of detail: "))

# Error checking.
if not isinstance(samples, int):
    raise Exception(
        "samples must be an integer. The value of detail_level was: {}".format(
            self.detail_level
        )
    )

if samples < 1:
    raise Exception(
        "samples must be a positive integer. The value of detail_level was: {}".format(
            self.detail_level
        )
    )

# -----------------------------------------------------------------------------------------#

# Benchmark function.
def benchmarking(samples):
    # CSV file.
    writer = csv_file()

    # Number of days in this month.
    number_of_days = amsimp.Backend.number_of_days

    # Loop by the number of samples.
    for i in range(samples):
        # Loop by each level of detail.
        for num in range(5):
            # Set level of detail.
            detail = amsimp.Weather(num + 1)

            # Start timer.
            start = time.time()

            # Random day of the month.
            random_day = random.randint(0, (number_of_days - 1))

            # Temperature, pressure thickness, precipitable water vapor, and
            # geostrophic wind on a random day of the month.
            temp = (
                detail.predict_temperature()[0] * random_day
            ) + detail.predict_temperature()[1]
            pressure_thickness = (
                detail.predict_pressurethickness()[0] * random_day
            ) + detail.predict_pressurethickness()[1]

            if (num + 1) >= 3:
                u_g = (
                    detail.predict_geostrophicwind()[0] * random_day
                ) + detail.predict_geostrophicwind()[1]
                P_wv = (
                    detail.predict_precipitablewater()[0] * random_day
                ) + detail.predict_precipitablewater()[1]

            # Store runtime in variable.
            finish = time.time()
            runtime = finish - start

            # Write runtime into CSV file.
            write_data(writer, {"detail_level": num + 1, "time": runtime})

            # Sample progress.
            if (num + 1) != 5:
                print(
                    "Progress of current sample: "
                    + str(int(((num + 1) / 5) * 100))
                    + "%"
                )

        # Overrall progress.
        if i != (samples - 1):
            print(
                "Progress: "
                + str(i + 1)
                + " samples have been run out of a total of "
                + str(samples)
            )

    # Benchmark complete.
    print("Benchmarking Complete.")


# -----------------------------------------------------------------------------------------#

benchmarking(samples)
