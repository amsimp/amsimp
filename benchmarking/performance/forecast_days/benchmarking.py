# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import amsimp
import time
from datetime import datetime
from datetime import timedelta
import csv
import os
import random

# -----------------------------------------------------------------------------------------#

# Output filename.
filename = input("The name of the CSV file: ")
filename = filename + ".csv"

# Creates a CSV file.
def csv_file():
    file = os.path.isfile(filename)
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

# Number of samples.
samples = int(input("The number of samples: "))

# Error checking.
if not isinstance(samples, int):
    raise Exception(
        "samples must be an integer. The value of samples was: {}".format(
            samples
        )
    )

if samples < 1:
    raise Exception(
        "samples must be a positive integer. The value of samples was: {}".format(
            samples
        )
    )

# -----------------------------------------------------------------------------------------#

# Benchmark function.
def benchmarking(samples):
    # CSV file.
    writer = csv_file()

    # Loop by the number of samples.
    for i in range(samples):
        s = time.time()

        # Loop by each the number of forecast days.
        for num in range(5):
            # Set the number of forecast days.
            detail = amsimp.Dynamics(5, (num + 1))

            # Start timer.
            start = time.time()

            detail.forecast_temperature()
            detail.forecast_pressure()
            detail.forecast_pressurethickness()
            detail.forecast_precipitablewater()

            # Store runtime in variable.
            finish = time.time()
            runtime = finish - start

            # Write runtime into CSV file.
            write_data(writer, {"forecast_days": num + 1, "time": runtime})

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
            f = time.time()
            r = f - s
            r *= ((samples - i) - 1)
            finish_time = datetime.now() + timedelta(seconds=+r)
            hour = finish_time.hour
            minute = finish_time.minute
            print(
                "Benchmarking will be finished at: "
                + str(hour) + ":" + str(minute)
            )

    # Benchmark complete.
    print("Benchmarking Complete.")


# -----------------------------------------------------------------------------------------#

benchmarking(samples)
